"""Tier 1 time integration and the scenario runner (src/time_integration.py, ADR-014 A, C, D).

ADR-014 G's step 4. What is pinned, each test naming in its docstring the
planted defect it was shown to catch:

- The plan: every interval divided into the fewest equal steps no larger than
  the smallest stable_dt over the classes on its faces, step i at t0 + i dt
  and the last at the interval's end exactly, so a switch lands on a step.
- The hand-over: at a switch the fields and the budgets are carried across
  unchanged into a transport solver on the new interval's faces, eddy
  viscosity and concentration boundary; the first step after a switch is
  reproduced bitwise by stepping the carried fields by hand.
- The sources: a dust release's emission enters the budget's source slot for
  its duration only, and the budget closes (REQ-T05).
- The three transport-only reference scenarios run end to end on a flow state
  kept in another run file; the door solves two states, the second
  warm-started from the first, and a second opening reuses the open state.
- The clean-out: section C's outputs, checked against the run file's own
  readings and frames; the tracker's arithmetic on a prescribed history.
- The restart: from a checkpoint after a switch, with no checkpoint, to a
  later t_end, and a capped flow solve continued under a larger cap, each the
  uninterrupted run bitwise where the ADR asks for it.
- scripts/run_scenario.py: its exit codes and summary lines, the alerts
  opened among them (ADR-015 C) and each sensor's latency per event, the
  monitor's beside the map's (REQ-A04).
- ADR-015 step 3, the arrival maps through the march: /arrival/ is section
  D's rule applied to every step's fields, bitwise; the record's two
  latencies agree to the step and are the stored series' rule; the frames'
  map is never before the step map and within one output interval where
  both reach a cell, and misses the cells at or above only between frames;
  a restart across a partly filled map is the uninterrupted run's; a
  checkpoint without the maps refuses the restart; a file without /arrival/
  is read from its frames and flagged; every run keeps the maps.

The coarse fixtures extend the reference scenario files with the grid
overridden to 80x30 and the error_estimate per-cell tolerance set by
ADR-011 G's formula at the base's t_end, the product row of record on that
grid (1021 outer iterations). A face hash of record belongs to one machine's
arithmetic, so no test here asserts one: the faces are compared with a solve
made in the same process, and the hashes of record are the harness's.
The event times of the transport-only and door fixtures are scaled down so
the suite marches seconds, not minutes; their segments, sources and initial
fields are the reference files'.
"""

import math
import re
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import run_scenario

from src import time_integration
from src.boundary_concentration import (
    ConcentrationBoundary,
)
from src.boundary_registry import BoundaryRegistry
from src.boundary_staggered import StaggeredBoundary
from src.config import SimConfig
from src.io_manager import RunFile
from src.mesh import SOLID, Mesh
from src.monitor import ARRIVAL_STATE_NAMES, Alert
from src.particles import ParticlePhysics
from src.placement import (
    FRAME_RESOLVED,
    STEP_RESOLVED,
    arrival_from_frames,
    arrival_from_run,
    detection_units,
)
from src.scenarios import Interval, ScenarioManager, flow_fingerprint
from src.solver_staggered import FlowState, StaggeredSolver
from src.solver_transport import TransportSolver
from src.staggered import FaceVelocities, allocate_fields
from src.time_integration import (
    DEFINED_RTOL,
    KEPT,
    REST,
    RESUMED,
    RUN_FILE,
    WARM,
    FlowNotConvergedError,
    MarchStep,
    Provenance,
    RecoveryTracker,
    _first_flow_difference,
    _Run,
    air_change_rate,
    flow_inputs,
    plan_interval,
    run,
    sensor_cells,
    smallest_stable_dt,
)
from tests.helpers import same_budgets, same_recovery
from validation.transport_cases import conservation_keys

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
COARSE_GRID = {"nx": 80, "ny": 30}
# ADR-011 G's per-cell bound, conservation_limit rho V_min / t_end, on 80x30 at the base's
# t_end of 60 s, as the product row of record on this grid ran it.
COARSE_MASS_IMBALANCE_TOL = (
    conservation_keys()["conservation_limit"] * 1.2 * (0.1 * 0.1) / 60.0
)
COARSE_OUTER = 1021
# The open door's state warm-started from the closed one on 80x30, as prompt
# 64 first solved it.
OPEN_OUTER = 928
# One air change of the slice, 1/N = 17.52 / 3.15 s (T6), and the t_end the
# held clean-out ran to under prompt 64.
ONE_AIR_CHANGE = 5.562
HELD_T_END = 120.0
# REQ-T05: particle mass conserved to 0.01%. The march closes to about 1e-7
# here, set by the implicit diffusion's diffusion_tol of 1e-8 per step.
REQ_T05 = conservation_keys()["conservation_limit"]
PROVENANCE = Provenance("test", "2026-10-17T00:00:00", "test-machine", 0)


class _KilledError(RuntimeError):
    """The run killed at a chosen step, as on_step raises it."""


def coarse_raw(name: str, **solver: object) -> dict:
    """A reference scenario file on the coarse grid, its base an absolute path."""
    raw = yaml.safe_load((CONFIGS / name).read_text(encoding="utf-8"))
    raw["scenario"]["base"] = str(CONFIGS / "clean_room_default.yaml")
    raw["domain"] = dict(COARSE_GRID)
    raw["solver"]["mass_imbalance_tol"] = COARSE_MASS_IMBALANCE_TOL
    raw["solver"].update(solver)
    return raw


def rescheduled(raw: dict, t_end: float, times: dict[str, tuple[float, float]]) -> dict:
    """RAW with t_end and the named events' (start, duration) replaced."""
    raw["solver"]["t_end"] = t_end
    for event in raw["scenario"]["events"]:
        if event["name"] in times:
            event["start"], event["duration"] = times[event["name"]]
    return raw


def door_events(raw: dict, second: tuple[float, float]) -> list:
    """The door's opening and a second one at SECOND = (start, duration)."""
    opening = raw["scenario"]["events"][0]
    again = yaml.safe_load(yaml.safe_dump(opening))
    again["name"] = "door_open_again"
    again["start"], again["duration"] = second
    return [opening, again]


def direct_solve(
    interval: Interval, mesh: Mesh, initial: FlowState | None = None
) -> FlowState:
    """An interval's flow state solved on its own, in this process, as the run solves it."""
    config = interval.config
    solver = StaggeredSolver(mesh, config, StaggeredBoundary(mesh, config))
    solver.solve_steady(initial=initial)
    assert solver.converged and solver.flow_state is not None
    return solver.flow_state


def same_faces(a: FlowState, b: FlowState) -> None:
    """Two states' face velocities, bitwise."""
    for name in ("u", "v"):
        mine = np.ascontiguousarray(getattr(a, name))
        theirs = np.ascontiguousarray(getattr(b, name))
        assert mine.shape == theirs.shape and mine.tobytes() == theirs.tobytes(), name


def recorder() -> tuple[list[MarchStep], Callable[[MarchStep], None]]:
    """A list and an on_step that appends to it."""
    steps: list[MarchStep] = []
    return steps, steps.append


def killer(at: int) -> Callable[[MarchStep], None]:
    """An on_step that kills the run after step AT's writes."""

    def kill(info: MarchStep) -> None:
        if info.step == at:
            raise _KilledError

    return kill


def same_frames(a: RunFile, b: RunFile) -> None:
    """Two run files' frames, budgets and sensor readings, bitwise."""
    fa, fb = a.read_frames(), b.read_frames()
    assert fa.steps.tolist() == fb.steps.tolist()
    assert fa.times.tobytes() == fb.times.tobytes()
    assert sorted(fa.fields) == sorted(fb.fields)
    for k in fb.fields:
        assert fa.fields[k].tobytes() == fb.fields[k].tobytes(), k
    for x, y in zip(fa.budgets, fb.budgets, strict=True):
        same_budgets(x, y)
    ra, rb = a.read_sensor_readings(), b.read_sensor_readings()
    assert ra.names == rb.names
    assert ra.steps.tolist() == rb.steps.tolist()
    assert ra.times.tobytes() == rb.times.tobytes()
    assert ra.readings.tobytes() == rb.readings.tobytes()


# ---------------------------------------------------------------------------
# The coarse runs, shared
# ---------------------------------------------------------------------------


def held_clean_out(**solver: object) -> dict:
    """The clean-out as prompt 64 ran it: to 120 s, held for one air change.

    The reference file runs to its t_end with no hold (prompt 66); the
    hold is a feature the suite pins on this run, which stops at about 31 s
    on 80x30.
    """
    raw = coarse_raw("scenario_clean_out.yaml", **solver)
    raw["solver"]["t_end"] = HELD_T_END
    raw["scenario"]["hold_time"] = ONE_AIR_CHANGE
    return raw


@pytest.fixture(scope="module")
def clean_out(tmp_path_factory: pytest.TempPathFactory) -> SimpleNamespace:
    """The coarse clean-out from rest to its hold: the kept base state for the rest.

    Checkpoints every 500 steps, which the hold's step (about 1770) is not
    a multiple of, so the hold's own checkpoint is pinned.
    """
    folder = tmp_path_factory.mktemp("clean_out")
    config = SimConfig.from_dict(held_clean_out(checkpoint_interval=500))
    record = run(config, folder / "run.h5", provenance=PROVENANCE)
    return SimpleNamespace(config=config, record=record, run=RunFile(record.path))


@pytest.fixture(scope="module")
def door(tmp_path_factory: pytest.TempPathFactory) -> SimpleNamespace:
    """The coarse door opened twice, both states solved, every step recorded.

    Open at 1 s for 1 s and again at 3 s for 0.5 s, to 5 s; checkpoints
    every 30 steps beside frames every 100, so the cadences do not divide.
    """
    folder = tmp_path_factory.mktemp("door")
    raw = coarse_raw("scenario_door_leak.yaml", checkpoint_interval=30)
    raw = rescheduled(raw, 5.0, {"door_open": (1.0, 1.0)})
    raw["scenario"]["events"] = door_events(raw, (3.0, 0.5))
    config = SimConfig.from_dict(raw)
    steps, on_step = recorder()
    record = run(config, folder / "run.h5", on_step=on_step, provenance=PROVENANCE)
    return SimpleNamespace(
        raw=raw, config=config, record=record, run=RunFile(record.path), steps=steps
    )


# ---------------------------------------------------------------------------
# Units: the plan, the sensors, the flow inputs
# ---------------------------------------------------------------------------


def _interval(t0: float, t1: float) -> Interval:
    return Interval(t0, t1, "0" * 64, {}, None, ())  # type: ignore[arg-type]


@pytest.mark.unit
class TestPlan:
    @pytest.mark.parametrize(
        ("t0", "t1", "stable", "steps"),
        [
            (0.0, 1.0, 0.3, 4),
            (10.0, 15.0, 0.017042023332377164, 294),
            (0.0, 0.25, 0.25, 1),
            (2.0, 2.5, 1.0, 1),
        ],
    )
    def test_the_fewest_equal_steps_no_larger(
        self, t0: float, t1: float, stable: float, steps: int
    ) -> None:
        """Defects caught: the step count rounded down (dt above stable_dt); one
        step too many; the plan read from the interval's start when t1 moved."""
        plan = plan_interval(3, _interval(t0, t1), stable)
        assert (plan.index, plan.steps) == (3, steps)
        assert plan.dt == (t1 - t0) / steps
        assert plan.dt <= stable
        assert (t1 - t0) / (steps - 1) > stable if steps > 1 else True
        assert plan.time(0) == t0 and plan.time(steps) == t1
        assert plan.time(1) == t0 + plan.dt

    def test_a_division_rounded_over_the_bound_takes_one_more_step(self) -> None:
        """0.07 / ceil(0.07 / 0.007) is 0.007000000000000001, above the bound.

        Defect caught: the ceiling taken as the count, so solve_timestep
        refuses the first step.
        """
        assert 0.07 / math.ceil(0.07 / 0.007) > 0.007
        plan = plan_interval(0, _interval(0.0, 0.07), 0.007)
        assert plan.steps == 11 and plan.dt <= 0.007

    def test_the_step_is_the_smallest_over_the_classes(self) -> None:
        """A stand-in whose classes' stable steps differ: the smallest is taken.

        On the coarse room every class's stable_dt is the same (the
        settling velocities do not move the worst cell's rate), so the run's
        plans cannot tell the smallest from the largest. Defects caught: the
        largest taken; the first class's alone.
        """
        steps = {0: 3.0, 1: 1.0, 2: 2.0}
        solver = SimpleNamespace(stable_dt=lambda faces, k: steps[k])
        assert smallest_stable_dt(solver, None, 3) == 1.0  # type: ignore[arg-type]

    def test_the_last_step_is_the_end_where_n_dt_misses_it(self) -> None:
        """10 + 167 (50 / 167) is 59.99999999999999; the last step is at 60 exactly.

        The door's intervals happen to give t0 + n dt == t1, so this case
        carries the pin. Defect caught: the last step's time computed as
        t0 + n dt, so the switch misses the boundary by an ulp.
        """
        plan = plan_interval(0, _interval(10.0, 60.0), 0.3)
        assert plan.steps == 167
        assert 10.0 + 167 * plan.dt != 60.0
        assert plan.time(167) == 60.0
        assert plan.time(166) == 10.0 + 166 * plan.dt

    def test_nothing_moving_is_one_step_and_a_start_moves_the_plan(self) -> None:
        """Defects caught: zero steps for an infinite stable_dt; t_start ignored."""
        assert plan_interval(0, _interval(0.0, 2.0), math.inf).steps == 1
        plan = plan_interval(1, _interval(1.0, 2.0), 0.3, t_start=1.25)
        assert (plan.t_start, plan.t_end, plan.steps) == (1.25, 2.0, 3)
        assert plan.dt == 0.75 / 3


@pytest.mark.unit
class TestSensorsAndInputs:
    def test_a_sensor_on_a_face_reads_the_cell_below_and_left_of_it(self) -> None:
        """The room's sensors sit on faces; the cell below or left of each is read.

        On 80x30 the nearest centre puts above_gap_1 (x 2.7) in column 27 by
        rounding and the rule puts it in 26; on 160x60 the rule reads the
        cells the probe of ADR-014 T7 read by the nearest centre. A sensor
        inside a cell reads that cell. Defects caught: the nearest centre
        kept (above_gap_1 in column 27); the tolerance dropped (a face that
        rounds below its sensor gives the upper cell); x and y swapped.
        """
        config = SimConfig.from_dict(coarse_raw("scenario_clean_out.yaml"))
        mesh = Mesh(config)
        cells = sensor_cells(config, mesh)
        assert cells == ((14, 9), (24, 26), (24, 47), (11, 61))
        assert int(np.argmin(np.abs(mesh.xc - 2.7))) == 27
        for (j, i), sensor in zip(cells, config.sensors, strict=True):
            assert sensor.x == pytest.approx(mesh.x[i + 1], abs=1e-12)
            assert sensor.y == pytest.approx(mesh.y[j + 1], abs=1e-12)
        raw = coarse_raw("scenario_clean_out.yaml")
        raw["domain"] = {"nx": 160, "ny": 60}
        raw["sensors"] = [
            *config.merged_mapping()["sensors"],
            {"name": "inside", "x": 0.33, "y": 0.77},
        ]
        fine = SimConfig.from_dict(raw)
        mesh = Mesh(fine)
        nearest = tuple(
            (
                int(np.argmin(np.abs(mesh.yc - s.y))),
                int(np.argmin(np.abs(mesh.xc - s.x))),
            )
            for s in fine.sensors
        )
        assert sensor_cells(fine, mesh) == nearest
        assert nearest[-1] == (15, 6)

    def test_on_the_product_grid_the_rule_reads_the_160x60_cells_faces(self) -> None:
        """On 320x120 each sensor's face is the 160x60 face, two cells finer.

        A face that is face i + 1 on 160x60 is face 2 (i + 1) on 320x120,
        so the cell below or left of it is 2 i + 1: the cell T7 read on
        160x60, halved. The nearest centre is a rounding tie on every
        sensor there. Defect caught: the rule claimed for 320x120 in
        ADR-014 C and not checked (review 64 S6).
        """
        cells = {}
        for grid in ((160, 60), (320, 120)):
            raw = coarse_raw("scenario_clean_out.yaml")
            raw["domain"] = {"nx": grid[0], "ny": grid[1]}
            config = SimConfig.from_dict(raw)
            mesh = Mesh(config)
            cells[grid] = sensor_cells(config, mesh)
            for (j, i), sensor in zip(cells[grid], config.sensors, strict=True):
                assert sensor.x == pytest.approx(mesh.x[i + 1], abs=1e-12)
                assert sensor.y == pytest.approx(mesh.y[j + 1], abs=1e-12)
        assert cells[(320, 120)] == tuple(
            (2 * j + 1, 2 * i + 1) for j, i in cells[(160, 60)]
        )

    def test_a_face_that_rounds_below_its_sensor_is_still_the_sensors_face(
        self,
    ) -> None:
        """On a stretched mesh a face can round just below the decimal it is written as.

        Face 3 of 80 cells stretched at 1.05 is 0.10438761230370124 and a
        sensor written to twelve places, 0.104387612304, lies above it by
        rounding. Defect caught: the tolerance dropped, so the sensor reads
        the cell to the right of its face.
        """
        raw = coarse_raw("scenario_clean_out.yaml")
        raw["mesh"] = {"x": {"stretch_ratio": 1.05}, "y": {"stretch_ratio": 1.0}}
        config = SimConfig.from_dict(raw)
        face = Mesh(config).x[3]
        written = round(float(face), 12)
        assert face < written
        raw["sensors"] = [{"name": "on_face", "x": written, "y": 1.5}]
        config = SimConfig.from_dict(raw)
        assert sensor_cells(config, Mesh(config)) == ((14, 2),)

    def test_a_sensor_in_a_solid_cell_is_refused(self) -> None:
        """Defect caught: a sensor reading a SOLID cell's zero as clean air."""
        raw = coarse_raw("scenario_clean_out.yaml")
        raw["sensors"] = [{"name": "inside_rack", "x": 1.9, "y": 1.0}]
        config = SimConfig.from_dict(raw)
        with pytest.raises(ValueError, match="sensor inside_rack reads a SOLID cell"):
            sensor_cells(config, Mesh(config))

    def test_the_flow_inputs_leave_out_what_the_flow_solve_does_not_read(
        self,
    ) -> None:
        """Defects caught: a run key, the transport section or the placement
        section (the product file's, prompt 75) counted, so a kept state is
        refused for a run that differs only in them; a solver key the solve
        reads left out, so a state of other settings is reused."""
        raw = SimConfig.from_dict(coarse_raw("scenario_clean_out.yaml"))
        mapping = raw.merged_mapping()
        inputs = flow_inputs(mapping)
        assert set(inputs) == {
            "domain",
            "mesh",
            "fluid",
            "solver",
            "turbulence",
            "obstacles",
        }
        left = set(mapping["solver"]) - set(inputs["solver"])
        assert left == {"dt", "t_end", "output_interval", "max_simple_iter"}
        assert inputs["solver"]["mass_imbalance_tol"] == COARSE_MASS_IMBALANCE_TOL
        assert mapping["solver"]["t_end"] == 240.0

    def test_a_solver_entry_that_is_not_a_mapping_passes_through(self) -> None:
        """Only a solver mapping is filtered; anything else is kept as given.

        The flow inputs are compared, not parsed, so an entry that is not a
        mapping must reach the comparison unchanged and not be iterated.
        Defect caught: the solver entry assumed to be a mapping (an
        AttributeError on a string).
        """
        assert flow_inputs({"domain": {"nx": 4}, "solver": "unset"}) == {
            "domain": {"nx": 4},
            "solver": "unset",
        }
        assert flow_inputs({"domain": {"nx": 4}}) == {"domain": {"nx": 4}}

    @pytest.mark.parametrize(
        ("kept", "ours", "expected"),
        [
            ({"a": {"x": 1}}, {"a": {"x": 1}}, None),
            ({"a": {"x": 1, "y": 2}}, {"a": {"x": 1, "y": 3}}, "a.y"),
            ({"a": {"x": 1}}, {"a": {"x": 1, "z": 0}}, "a.z"),
            ({"a": {"x": 1, "y": None}}, {"a": {"x": 1}}, "a.y"),
            ({"obstacles": [1, 2]}, {"obstacles": [1]}, "obstacles"),
            ({"a": {"x": 1}}, {"a": 5}, "a"),
            ({"a": 1}, {"a": 1, "extra": {"k": 1}}, "extra"),
            ({"a": 1, "b": 2}, {"a": 9, "b": 8}, "a"),
        ],
    )
    def test_the_first_flow_difference_names_the_section_or_the_key(
        self, kept: dict, ours: dict, expected: str | None
    ) -> None:
        """A mapping section is named down to its first differing key.

        A section that is not a mapping on both sides, or that exists on one
        side only, is named whole, the sections in the kept file's order. A
        key present with value None on one side only is a difference.
        Defects caught: a section named whole when its key differs; a key
        present on one side with value None passed over; a section found in
        the new file only skipped.
        """
        assert _first_flow_difference(kept, ours) == expected


# ---------------------------------------------------------------------------
# Units: the recovery tracker on a prescribed history
# ---------------------------------------------------------------------------

# A 1 by 4 grid, column 2 SOLID, sensors on columns 0 and 3. Class 0 starts
# at 100 times its threshold (the 100:1 times defined), class 1 at 50 times
# (not defined). Each row: the class 0 field, the class 1 field, the means.
LIVE = np.array([[True, True, False, True]])
HISTORY = [
    ([1000.0, 1000.0, 0.0, 1000.0], [50.0, 50.0, 0.0, 0.5], [1000.0, 50.0]),
    ([50.0, 20.0, 0.0, 9.0], [0.5, 0.5, 0.0, 0.5], [30.0, 0.6]),
    ([9.0, 12.0, 0.0, 8.0], [0.4, 0.2, 0.0, 0.3], [9.5, 0.3]),
    ([11.0, 5.0, 0.0, 7.0], [0.3, 0.2, 0.0, 0.2], [11.0, 0.25]),
    ([10.0, 4.0, 0.0, 3.0], [0.2, 0.3, 0.0, 0.1], [10.0, 0.2]),
    ([2.0, 3.0, 0.0, 1.0], [0.1, 0.5, 0.0, 0.1], [2.0, 0.2]),
    ([1.0, 2.0, 0.0, 0.5], [0.1, 1.0, 0.0, 0.1], [1.0, 0.4]),
]
RATE = math.log(100.0) / 4.0


def _tracker(hold: float | None = None) -> RecoveryTracker:
    return RecoveryTracker([10.0, 1.0], [1000.0, 50.0], [(0, 0), (0, 3)], LIVE, hold)


def _feed(tracker: RecoveryTracker, rows: list) -> list[bool]:
    """Update with each row at t = its index; the held() after each."""
    held = []
    for n, (c0, c1, means) in rows:
        tracker.update(
            float(n), {0: np.array([c0]), 1: np.array([c1])}, np.array(means)
        )
        held.append(tracker.held())
    return held


@pytest.mark.unit
class TestRecoveryTracker:
    def test_each_output_on_the_prescribed_history(self) -> None:
        """Every output by hand.

        Class 0 is within spec at t = 2 (9 and 8), out at 3 (11) and at 4
        (10 is not below 10), and in from 5: recovered at 5. Class 1 is in
        from 1. The room is in at 2 and from 5: recovered at 5. Sensor 1 of
        class 0 first reads at most 10 at t = 1, sensor 0 at t = 2, the mean
        at t = 2 (9.5; the 10 at t = 4 is later). Class 1's 100:1 times are
        not defined though its readings fall to 0.5. The local field: class
        0's cells last at or above 10 at 4, 2 and 0 (the initial field
        only); class 1's east cell starts below its threshold and is never
        above it, so zero, and its middle cell is at its threshold at the
        end, so not recovered. Defects caught (one
        per plant): the within test at or below rather than below; a stretch
        not reset when a sensor goes back over; the room within when any
        class is; a 100:1 time overwritten by a later crossing; the times
        reported where not defined; the local field's above strict, its
        never-above cells NaN or its still-above cells finite; the ratio over
        N rather than ln(100)/N; the maximum taken over SOLID cells.
        """
        tracker = _tracker()
        _feed(tracker, list(enumerate(HISTORY)))
        record = tracker.record(["west", "east"], RATE, held=False)
        assert record.room_recovery_time == 5.0
        assert record.recovery_time.tolist() == [5.0, 1.0]
        assert record.defined.tolist() == [True, False]
        assert record.sensor_time[0].tolist() == [2.0, 1.0]
        assert np.isnan(record.sensor_time[1]).all()
        assert record.mean_time[0] == 2.0 and np.isnan(record.mean_time[1])
        assert record.well_mixed_time == 4.0
        assert record.ratio[0] == 0.5 and np.isnan(record.ratio[1])
        local = record.local_recovery
        assert local[0, 0, [0, 1, 3]].tolist() == [4.0, 2.0, 0.0]
        assert local[1, 0, [0, 1, 3]].tolist() == [0.0, math.inf, 0.0]
        assert np.isnan(local[:, 0, 2]).all()
        assert record.local_recovery_max.tolist() == [4.0, math.inf]
        assert (record.t_final, record.held, record.hold_time) == (6.0, False, None)
        assert record.thresholds.tolist() == [10.0, 1.0]
        assert record.initial.tolist() == [1000.0, 50.0]

    @pytest.mark.parametrize(
        ("hold", "held", "room", "classes"),
        [
            (1.0, [False] * 6 + [True], 5.0, [5.0, 1.0]),
            (1.5, [False] * 7, math.nan, [math.nan, 1.0]),
            (5.0, [False] * 7, math.nan, [math.nan, 1.0]),
        ],
    )
    def test_the_hold(
        self, hold: float, held: list[bool], room: float, classes: list[float]
    ) -> None:
        """held() turns true once the room has been in spec for the hold; a stretch
        shorter than it recovers nothing.

        Defects caught: the hold compared with > (a hold of exactly one air
        change never ends); held() measured from the first within-spec step
        rather than the current stretch's; the recovery reported for a
        stretch shorter than the hold.
        """
        tracker = _tracker(hold)
        assert _feed(tracker, list(enumerate(HISTORY))) == held
        record = tracker.record(["west", "east"], RATE, held=held[-1])
        np.testing.assert_array_equal(record.room_recovery_time, room)
        np.testing.assert_array_equal(record.recovery_time, classes)
        assert record.hold_time == hold

    def test_the_100_to_1_time_is_defined_to_a_tolerance(self) -> None:
        """35.2 x 100 is 3520.0000000000005; a stated 3520 is still at 100 thresholds.

        Defects caught: the comparison exact, so a clean-out stated by
        concentration has no 100:1 time; the tolerance applied the wrong way.
        """
        live = np.ones((1, 1), dtype=bool)
        cells = [(0, 0)]
        assert 35.2 * 100.0 > 3520.0
        defined = RecoveryTracker([35.2], [3520.0], cells, live, None)._defined
        assert defined.tolist() == [True]
        below = 3520.0 * (1.0 - 10.0 * DEFINED_RTOL)
        defined = RecoveryTracker([35.2], [below], cells, live, None)._defined
        assert defined.tolist() == [False]

    def test_a_restored_tracker_continues_bitwise(self) -> None:
        """State taken after t = 3, restored in a new tracker, fed the rest.

        Defects caught: an array left out of state(); restore keeping the
        new tracker's arrays; the time not carried, so held() is wrong.
        """
        whole = _tracker(1.0)
        _feed(whole, list(enumerate(HISTORY)))
        first = _tracker(1.0)
        _feed(first, list(enumerate(HISTORY))[:4])
        later = _tracker(1.0)
        later.restore({k: np.array(v) for k, v in first.state().items()})
        assert _feed(later, list(enumerate(HISTORY))[4:]) == [False, False, True]
        same_recovery(
            later.record(["w", "e"], RATE, True), whole.record(["w", "e"], RATE, True)
        )

    def test_bad_construction_and_restores_are_refused(self) -> None:
        """Defects caught: a clean-out with no sensor accepted (the room within
        spec at t = 0 for want of a reading); a state of another shape restored."""
        with pytest.raises(ValueError, match="needs at least one sensor"):
            RecoveryTracker([1.0], [100.0], [], LIVE, None)
        with pytest.raises(ValueError, match="2 thresholds and 1 initial"):
            RecoveryTracker([1.0, 2.0], [100.0], [(0, 0)], LIVE, None)
        tracker = _tracker()
        state = tracker.state()
        state["last_above"] = np.zeros((2, 1, 5))
        with pytest.raises(ValueError, match="last_above has shape"):
            tracker.restore(state)
        del state["last_above"]
        with pytest.raises(KeyError):
            tracker.restore(state)


@pytest.mark.unit
class TestRecoveryLines:
    """run_scenario.recovery_lines on records the prescribed history makes."""

    def test_the_held_not_defined_and_ratio_lines(self) -> None:
        """Held after a hold of 1 s; class 1 not defined; class 0's ratio 0.5.

        The record is TestRecoveryTracker's history, whose every output is
        checked by hand there. Defects caught: the held line, the not-defined
        line or the ratio line dropped or fed another field (the mean's time
        for the ratio, a sensor's time for the mean), or the classes' lines
        transposed (review 64 S9).
        """
        tracker = _tracker(1.0)
        assert _feed(tracker, list(enumerate(HISTORY)))[-1]
        record = tracker.record(["west", "east"], RATE, held=True)
        lines = run_scenario.recovery_lines(record, [1.0e-6, 5.0e-6])
        assert lines == [
            "recovery (ADR-014 C):",
            f"  air change rate N = {RATE:.6g} 1/s, well-mixed 100:1 time "
            "ln(100)/N = 4.0000 s",
            "  stopped by the hold of 1.0 s",
            "  room recovery time: 5.0000 s",
            "  class 0 (1 um):",
            "    recovery time: 5.0000 s",
            "    free-area mean 100:1: 2.0000 s, ratio to well-mixed 0.5000",
            "    sensor 100:1: west 2.0000 s, east 1.0000 s",
            "    local recovery field maximum: 4.0000 s",
            "  class 1 (5 um):",
            "    recovery time: 1.0000 s",
            "    100:1 times: not defined (initial below 100 thresholds)",
            "    local recovery field maximum: not recovered (a cell still above)",
        ]


# ---------------------------------------------------------------------------
# The run's guards
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRunGuards:
    def test_what_run_refuses_before_anything_is_solved(self, tmp_path: Path) -> None:
        """Defects caught: a configuration without a scenario marched as one
        interval; a new run appended to an existing file or beside a stale
        checkpoint file; kept states mixed into a restart."""
        plain = SimConfig(CONFIGS / "clean_room_default.yaml")
        with pytest.raises(ValueError, match="no scenario section"):
            run(plain, tmp_path / "a.h5")
        config = SimConfig.from_dict(coarse_raw("scenario_clean_out.yaml"))
        with pytest.raises(ValueError, match="exactly one of run_file"):
            run(config)
        with pytest.raises(ValueError, match="exactly one of run_file"):
            run(config, tmp_path / "a.h5", restart=tmp_path / "b.h5")
        with pytest.raises(ValueError, match="flows_from is for a new run"):
            run(config, restart=tmp_path / "b.h5", flows_from=tmp_path / "c.h5")
        u, v, p = allocate_fields(Mesh(config))
        with pytest.raises(ValueError, match="initial is tier 2's"):
            run(config, tmp_path / "a.h5", initial=FlowState(u=u, v=v, p=p))
        for existing in (tmp_path / "x.h5", tmp_path / "y.checkpoint.h5"):
            existing.write_bytes(b"")
        for name in ("x.h5", "y.h5"):
            with pytest.raises(ValueError, match="exists; a new run writes new files"):
                run(config, tmp_path / name)
        assert not (tmp_path / "a.h5").exists()

    def test_a_checkpoint_past_every_interval_is_refused(self, tmp_path: Path) -> None:
        """``resume_point`` names a checkpoint time that lies in no plan.

        ``run`` refuses a checkpoint at or past t_end first, so this is the
        guard behind that one. Defect caught: a time past the plans resumed
        in the last interval, which would march a run beyond its end.
        """
        config = SimConfig.from_dict(coarse_raw("scenario_clean_out.yaml"))
        state = _Run(config, RunFile(tmp_path / "x.h5"), None)
        plans = [plan_interval(i, iv, 0.1) for i, iv in enumerate(state.intervals)]
        before, past = plans[0].t_start, plans[-1].t_end
        assert state.resume_point(plans, before, None)[1] == (0, 0)
        for t in (past, past + 1.0):
            with pytest.raises(AssertionError, match="lies in no interval"):
                state.resume_point(plans, t, None)


@pytest.mark.integration
class TestRunGuardsOnAKeptFile:
    """The guard on flows_from, which needs a kept file: the clean-out's solve."""

    @pytest.mark.parametrize(
        ("section", "key", "value", "named"),
        [
            ("domain", "nx", 40, "domain.nx"),
            ("solver", "alpha_velocity", 0.4, "solver.alpha_velocity"),
            ("turbulence", "variant", "rng", "turbulence.variant"),
        ],
    )
    def test_a_kept_file_of_other_flow_inputs_is_refused(
        self,
        tmp_path: Path,
        clean_out: SimpleNamespace,
        section: str,
        key: str,
        value: object,
        named: str,
    ) -> None:
        """Defects caught: the kept file's configuration not compared, so a state of
        another grid or other solver settings is reused under its fingerprint."""
        raw = coarse_raw("scenario_equipment_dust.yaml")
        raw.setdefault(section, {})[key] = value
        config = SimConfig.from_dict(raw)
        with pytest.raises(ValueError, match=rf"flows_from: {named} differs"):
            run(config, tmp_path / "run.h5", flows_from=clean_out.record.path)
        assert not (tmp_path / "run.h5").exists()


# ---------------------------------------------------------------------------
# The clean-out
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestCleanOut:
    def test_the_state_is_the_coarse_product_row_of_record(
        self, clean_out: SimpleNamespace
    ) -> None:
        """One state solved from rest, the faces a direct solve's bitwise.

        The reference is a solve_steady of the same configuration made here,
        not a hash recorded elsewhere. Defects caught: the first state
        warm-started from something; the run file holding a state other
        than the solve's.
        """
        record = clean_out.record
        assert [f.origin for f in record.flows] == [REST]
        flow = record.flows[0]
        assert (flow.outer, flow.stop_reason) == (
            COARSE_OUTER,
            "error_estimate_and_continuity",
        )
        state = clean_out.run.read_flow_state(flow.fingerprint).state
        mesh = Mesh(clean_out.config)
        first = ScenarioManager(clean_out.config, mesh).intervals()[0]
        direct = direct_solve(first, mesh)
        assert direct.outer == COARSE_OUTER
        same_faces(state, direct)
        assert clean_out.run.flow_fingerprints() == (flow.fingerprint,)
        assert air_change_rate(state, Mesh(clean_out.config)) == pytest.approx(
            3.15 / 17.52, rel=1e-12
        )

    def test_the_hold_stops_the_run_and_the_outputs_are_written(
        self, clean_out: SimpleNamespace
    ) -> None:
        """Held, the room recovered a hold before the end, the record in the file.

        Defects caught: the hold not checked (the run goes to t_end); the
        outputs not written, or written before the last step.
        """
        record = clean_out.record
        recovery = record.recovery
        assert record.held and recovery.held
        assert record.t_final < clean_out.config.t_end
        assert recovery.t_final == record.t_final
        hold = clean_out.config.scenario.hold_time
        assert recovery.hold_time == hold
        assert record.t_final - recovery.room_recovery_time >= hold
        plan = record.intervals[0]
        assert record.t_final - plan.dt - recovery.room_recovery_time < hold
        assert recovery.room_recovery_time == np.max(recovery.recovery_time)
        same_recovery(clean_out.run.read_recovery(), recovery)

    def test_the_recovery_times_are_the_run_files_readings(
        self, clean_out: SimpleNamespace
    ) -> None:
        """Recomputed from /transport/sensors/: every step from 0, no gap.

        Each class's recovery time is the first step of the trailing run of
        steps whose every sensor reads below its threshold; each sensor's
        100:1 time is its first reading at or below C0 / 100. Defects caught:
        a reading lost at a checkpoint or a frame; the tracker reading other
        cells than the file records.
        """
        readings = clean_out.run.read_sensor_readings()
        record = clean_out.record
        assert readings.steps.tolist() == list(range(record.steps + 1))
        assert readings.names == tuple(s.name for s in clean_out.config.sensors)
        recovery = record.recovery
        thresholds = np.array(clean_out.config.class_thresholds())
        within = (readings.readings < thresholds[None, :, None]).all(axis=2)
        for k in range(len(thresholds)):
            assert within[-1, k]
            start = (
                len(within) - np.argmin(within[::-1, k])
                if not within[:, k].all()
                else 0
            )
            assert recovery.recovery_time[k] == readings.times[start], k
            for s in range(len(readings.names)):
                hit = np.flatnonzero(
                    readings.readings[:, k, s] <= recovery.initial[k] / 100.0
                )
                assert recovery.sensor_time[k, s] == readings.times[hit[0]], (k, s)

    def test_the_local_field_and_the_mean_against_the_last_frame(
        self, clean_out: SimpleNamespace
    ) -> None:
        """inf exactly where the last frame is at or above the threshold.

        The free-area mean has a 100:1 time only if the last frame's mean is
        at or below C0 / 100 (or it fell there earlier). Defects caught: the
        still-above cells read from an earlier step; a mean time where the
        mean never fell; the field written with SOLID cells as zero.
        """
        record = clean_out.record
        frames = clean_out.run.read_frames()
        assert frames.steps[-1] == record.steps
        mesh = Mesh(clean_out.config)
        live = mesh.cell_type != SOLID
        recovery = record.recovery
        for k, threshold in enumerate(recovery.thresholds):
            last = frames.fields[k][-1]
            local = recovery.local_recovery[k]
            assert np.array_equal(np.isinf(local), live & (last >= threshold))
            assert np.isnan(local[~live]).all()
            finite = local[live & ~np.isinf(local)]
            assert ((finite >= 0.0) & (finite <= record.t_final)).all()
            mean = frames.budgets[-1][k].current / float(
                np.outer(mesh.dy_cell, mesh.dx_cell)[live].sum()
            )
            if np.isnan(recovery.mean_time[k]):
                assert mean > recovery.initial[k] / 100.0
            else:
                assert recovery.ratio[k] == recovery.mean_time[k] / (
                    recovery.well_mixed_time
                )

    def test_the_hold_writes_its_checkpoint_and_a_restart_returns_held(
        self, tmp_path: Path, clean_out: SimpleNamespace
    ) -> None:
        """The hold's step is a frame and a checkpoint; restarted, the run is held.

        A copy of the held run's files restarted under the same
        configuration continues from the hold's checkpoint, takes no step and
        is held, its frames, readings and recovery outputs unchanged.
        Defects caught: the hold's step not treated as the last (no
        checkpoint there, so the restart marches on from step 1500); the
        restored hold not checked before the march (the restart steps past
        the hold); the hold's recovery rewritten as not held (review 64
        S4.3).
        """
        record = clean_out.record
        cadence = clean_out.config.checkpoint_interval
        assert record.held and record.steps % cadence != 0
        checkpoint = clean_out.run.read_checkpoint()
        assert (checkpoint.step, checkpoint.t) == (record.steps, record.t_final)
        path = tmp_path / "run.h5"
        _copy_run(record.path, path)
        steps, on_step = recorder()
        again = run(clean_out.config, restart=path, on_step=on_step)
        assert steps == []
        assert again.restarted_from == record.steps
        assert (again.steps, again.t_final, again.held) == (
            record.steps,
            record.t_final,
            True,
        )
        same_frames(RunFile(path), clean_out.run)
        same_recovery(again.recovery, record.recovery)
        same_recovery(RunFile(path).read_recovery(), record.recovery)

    def test_frames_and_budgets(self, clean_out: SimpleNamespace) -> None:
        """Frames at 0, every output_interval and the last step; budgets close.

        Defects caught: the last frame not written at a hold; a budget
        restarted at the switch-free start; closure outside REQ-T05.
        """
        record = clean_out.record
        frames = clean_out.run.read_frames()
        every = clean_out.config.output_interval
        expected = [*range(0, record.steps, every), record.steps]
        assert frames.steps.tolist() == sorted(set(expected))
        for budget in record.budgets:
            assert abs(budget.relative()) < REQ_T05
        same_budgets(frames.budgets[-1], record.budgets)
        assert all((c >= 0.0).all() for c in frames.fields.values())

    def test_the_tracker_reads_each_steps_fields_and_free_area_means(
        self,
        tmp_path: Path,
        clean_out: SimpleNamespace,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """What the march hands the tracker, step by step, against on_step.

        The free-area mean of the coarse clean-out falls to its 100:1 level
        only after about 74 s (the coarse no-hold run, results/builder64/
        coarse), beyond any run the suite can afford, so its arithmetic is
        checked where it is handed over: each update's time, fields and means
        are the step's, the means the budget's content over the non-SOLID
        area. Defects caught: the content handed over undivided, or the
        field's sum unweighted by the cell areas; the fields of the step
        before.
        """
        handed: list[tuple[float, dict, np.ndarray]] = []
        update = RecoveryTracker.update

        def spy(self: RecoveryTracker, t: float, fields: dict, means: object) -> None:
            handed.append((t, dict(fields), np.array(means)))
            update(self, t, fields, means)

        monkeypatch.setattr(RecoveryTracker, "update", spy)
        raw = rescheduled(coarse_raw("scenario_clean_out.yaml"), 6.0, {})
        steps, on_step = recorder()
        run(
            SimConfig.from_dict(raw),
            tmp_path / "run.h5",
            flows_from=clean_out.record.path,
            on_step=on_step,
            provenance=PROVENANCE,
        )
        mesh = Mesh(clean_out.config)
        live = mesh.cell_type != SOLID
        area = np.outer(mesh.dy_cell, mesh.dx_cell)
        free = float(area[live].sum())
        assert len(handed) == len(steps) + 1
        t0, fields0, means0 = handed[0]
        assert t0 == 0.0
        for k, c in fields0.items():
            assert means0[k] == float(np.sum((c * area)[live])) / free
        for (t, fields, means), info in zip(handed[1:], steps, strict=True):
            assert t == info.t
            for k, c in info.fields.items():
                assert fields[k].tobytes() == c.tobytes()
                assert means[k] == info.budgets[k].current / free
                assert means[k] == pytest.approx(
                    float(np.sum(c * area * live)) / free, rel=1e-12
                )


# ---------------------------------------------------------------------------
# The three transport-only scenarios on a kept state
# ---------------------------------------------------------------------------

TRANSPORT_ONLY = {
    "scenario_equipment_dust.yaml": (4.0, {"litho_dust": (1.0, 2.0)}),
    "scenario_filter_breach.yaml": (4.0, {"panel_breach": (1.0, 2.0)}),
    "scenario_clean_out.yaml": (6.0, {}),
}


@pytest.fixture(scope="module")
def transport_only(
    tmp_path_factory: pytest.TempPathFactory, clean_out: SimpleNamespace
) -> dict[str, SimpleNamespace]:
    """Each transport-only scenario, rescheduled, on the clean-out's kept state."""
    out = {}
    for name, (t_end, times) in TRANSPORT_ONLY.items():
        folder = tmp_path_factory.mktemp(name.removesuffix(".yaml"))
        raw = rescheduled(coarse_raw(name, output_interval=10), t_end, times)
        config = SimConfig.from_dict(raw)
        record = run(
            config,
            folder / "run.h5",
            flows_from=clean_out.record.path,
            provenance=PROVENANCE,
        )
        out[name] = SimpleNamespace(
            config=config, record=record, run=RunFile(record.path)
        )
    return out


@pytest.mark.integration
class TestTransportOnly:
    @pytest.mark.parametrize("name", sorted(TRANSPORT_ONLY))
    def test_end_to_end_on_the_kept_state(
        self,
        name: str,
        transport_only: dict[str, SimpleNamespace],
        clean_out: SimpleNamespace,
    ) -> None:
        """One state, kept not solved, copied bitwise; every interval marched; closed.

        Defects caught: a kept state solved again; a transport-only event
        adding a state; the run stopping short of t_end; the copied state
        differing from the kept one.
        """
        case = transport_only[name]
        record = case.record
        kept = clean_out.record.flows[0]
        assert [(f.fingerprint, f.origin, f.solved) for f in record.flows] == [
            (kept.fingerprint, KEPT, False)
        ]
        assert {p.fingerprint for p in record.intervals} == {kept.fingerprint}
        mine = case.run.read_flow_state(kept.fingerprint).state
        theirs = clean_out.run.read_flow_state(kept.fingerprint).state
        for field in ("u", "v", "p", "k", "eps", "nu_t"):
            assert getattr(mine, field).tobytes() == getattr(theirs, field).tobytes()
        n_events = len(TRANSPORT_ONLY[name][1])
        assert len(record.intervals) == 1 + 2 * n_events
        held = record.held
        assert (record.t_final == case.config.t_end) != held
        assert record.steps == sum(p.steps for p in record.intervals) or held
        for budget in record.budgets:
            assert abs(budget.relative()) < REQ_T05
        frames = case.run.read_frames()
        assert all((c >= 0.0).all() for c in frames.fields.values())

    def test_a_run_without_sensors_writes_its_frames_and_no_readings(
        self, tmp_path: Path, clean_out: SimpleNamespace
    ) -> None:
        """An empty sensor list is a run: frame 0 and the march, no readings group.

        Defects caught: the readings of zero sensors written as an empty
        array (a group with no columns), a run refused for having no sensor,
        and frame 0 skipped with the readings.
        """
        raw = rescheduled(
            coarse_raw("scenario_equipment_dust.yaml", output_interval=10),
            2.0,
            {"litho_dust": (0.5, 1.0)},
        )
        raw["sensors"] = []
        config = SimConfig.from_dict(raw)
        record = run(
            config,
            tmp_path / "run.h5",
            flows_from=clean_out.record.path,
            provenance=PROVENANCE,
        )
        written = RunFile(record.path)
        frames = written.read_frames()
        assert frames.steps[0] == 0 and frames.steps.size > 1
        assert record.steps == frames.steps[-1] or record.steps > frames.steps[-1]
        with pytest.raises(KeyError, match="holds no /transport/sensors/"):
            written.read_sensor_readings()

    @pytest.mark.parametrize(
        ("name", "match"),
        [
            ("scenario_equipment_dust.yaml", "the alert monitor: the class of diam"),
            ("scenario_clean_out.yaml", "the clean-out's recovery: the class of diam"),
        ],
    )
    def test_sensors_without_a_class_threshold_refuse_the_run(
        self, tmp_path: Path, name: str, match: str
    ) -> None:
        """A run with sensors needs every class's threshold; a clean-out names its recovery.

        The last class is given a diameter the thresholds have no key for (a
        scenario's thresholds merge over the base's, so a key cannot be
        removed), and the clean-out's initial is stated as a concentration,
        since a threshold factor needs the thresholds at load. The monitor is
        built after the tracker, so the clean-out is refused for its recovery
        first (review 73 S6). Defects caught: the run starting without a
        class's threshold; the monitor's message given to a clean-out.
        """
        raw = rescheduled(coarse_raw(name), 2.0, {"litho_dust": (0.5, 1.0)})
        raw["particles"] = {"sizes": [0.1e-6, 0.3e-6, 0.5e-6, 1.0e-6, 2.0e-6]}
        for event in raw["scenario"]["events"]:
            if "initial" in event:
                event["initial"] = {"concentration": [1.0e7] * 5}
        config = SimConfig.from_dict(raw)
        with pytest.raises(ValueError, match=match):
            run(config, tmp_path / "run.h5", provenance=PROVENANCE)

    def test_a_run_without_sensors_restarts_with_no_monitor(
        self, tmp_path: Path, clean_out: SimpleNamespace
    ) -> None:
        """Killed after a checkpoint and restarted: the frames bitwise, still no alerts.

        A run without sensors has no monitor, so the restart has none to
        restore. Defects caught: the restart restoring a monitor the run never
        had; an alert table written for a run without sensors.
        """
        raw = rescheduled(
            coarse_raw(
                "scenario_equipment_dust.yaml",
                output_interval=10,
                checkpoint_interval=20,
            ),
            2.0,
            {"litho_dust": (0.5, 1.0)},
        )
        raw["sensors"] = []
        config = SimConfig.from_dict(raw)
        whole = run(
            config,
            tmp_path / "whole.h5",
            flows_from=clean_out.record.path,
            provenance=PROVENANCE,
        )
        path = tmp_path / "run.h5"
        with pytest.raises(_KilledError):
            run(
                config,
                path,
                flows_from=clean_out.record.path,
                on_step=killer(43),
                provenance=PROVENANCE,
            )
        record = run(config, restart=path)
        assert record.restarted_from == 40 and record.alerts == ()
        a, b = RunFile(path).read_frames(), RunFile(whole.path).read_frames()
        assert a.steps.tolist() == b.steps.tolist()
        for k in b.fields:
            assert a.fields[k].tobytes() == b.fields[k].tobytes(), k
        with pytest.raises(KeyError, match="holds no /transport/alerts/"):
            RunFile(path).read_alerts()

    def test_the_source_enters_the_source_slot_for_its_duration(
        self, transport_only: dict[str, SimpleNamespace]
    ) -> None:
        """The dust release books 1e4 per second per metre per class from 1 s to 3 s.

        Read from the budgets of every frame (every 10 steps): zero up to the
        start, the emission times the time since the start during it, the
        emission times the duration after. Defects caught: the sources added
        outside their interval; the sources passed for one class only; the
        rate read at the wrong time; the sources left out of the budget.
        """
        case = transport_only["scenario_equipment_dust.yaml"]
        frames = case.run.read_frames()
        start, duration = TRANSPORT_ONLY["scenario_equipment_dust.yaml"][1][
            "litho_dust"
        ]
        emission = 1.0e4
        for t, budgets in zip(frames.times, frames.budgets, strict=True):
            booked = emission * min(max(t - start, 0.0), duration)
            for budget in budgets:
                assert budget.source == pytest.approx(booked, rel=1e-12, abs=1e-9)
        for budget in case.record.budgets:
            assert budget.source == pytest.approx(emission * duration, rel=1e-12)

    def test_the_breach_carries_air_in_only_while_it_lasts(
        self, transport_only: dict[str, SimpleNamespace]
    ) -> None:
        """The panel's concentration enters from 1 s to 3 s and not before.

        Defects caught: the interval's concentration boundary not rebuilt
        at the switch (no inflow, or inflow to the end).
        """
        case = transport_only["scenario_filter_breach.yaml"]
        frames = case.run.read_frames()
        inflow = np.array([[b.inflow for b in budgets] for budgets in frames.budgets])
        before = frames.times <= 1.0
        after = frames.times >= 3.0
        assert (inflow[before] == 0.0).all()
        assert (inflow[after] == inflow[after][0]).all()
        assert (inflow[after][0] > 0.0).all()


# ---------------------------------------------------------------------------
# The door: two states, warm start, reuse; the plan; the hand-over
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestDoor:
    def test_two_states_solved_the_second_warm_and_reused(
        self, door: SimpleNamespace
    ) -> None:
        """Closed from rest, open warm from closed, the second opening solved by nobody.

        The references are solves made here: the closed state from rest and
        the open one warm-started from it, the run's faces bitwise and its
        outer counts equal. Defects caught: the second opening solved again;
        the open state started from rest (its faces and count are the rest
        solve's, though the label says warm); a state warm-started from a
        state other than the one before it.
        """
        record = door.record
        closed, opened = (f.fingerprint for f in record.flows)
        assert [p.fingerprint for p in record.intervals] == [
            closed,
            opened,
            closed,
            opened,
            closed,
        ]
        assert [(f.origin, f.warm_from) for f in record.flows] == [
            (REST, None),
            (WARM, closed),
        ]
        assert all(
            f.stop_reason == "error_estimate_and_continuity" for f in record.flows
        )
        assert [f.outer for f in record.flows] == [COARSE_OUTER, OPEN_OUTER]
        mesh = Mesh(door.config)
        intervals = ScenarioManager(door.config, mesh).intervals()
        direct_closed = direct_solve(intervals[0], mesh)
        direct_open = direct_solve(intervals[1], mesh, initial=direct_closed)
        assert (direct_closed.outer, direct_open.outer) == (COARSE_OUTER, OPEN_OUTER)
        same_faces(door.run.read_flow_state(closed).state, direct_closed)
        same_faces(door.run.read_flow_state(opened).state, direct_open)
        assert sorted(door.run.flow_fingerprints()) == sorted([closed, opened])
        assert [p.events for p in record.intervals] == [
            (),
            ("door_open",),
            (),
            ("door_open_again",),
            (),
        ]

    def test_every_interval_is_divided_and_every_switch_is_a_step(
        self, door: SimpleNamespace
    ) -> None:
        """Every step's time is its plan's, the last on the interval's end exactly.

        stable_dt is recomputed per state from the run file's faces, the
        smallest over the classes. Defects caught: the time accumulated
        (the switch misses the boundary by rounding); stable_dt kept from
        the first state; the largest over the classes taken.
        """
        record = door.record
        steps = door.steps
        assert [s.step for s in steps] == list(range(1, record.steps + 1))
        mesh = Mesh(door.config)
        manager = ScenarioManager(door.config, mesh)
        intervals = manager.intervals()
        n = 0
        for plan, interval in zip(record.intervals, intervals, strict=True):
            mine = steps[n : n + plan.steps]
            n += plan.steps
            assert all(s.interval == plan.index for s in mine)
            assert [s.t for s in mine] == [
                plan.t_start + i * plan.dt for i in range(1, plan.steps)
            ] + [interval.t_end]
            assert all(s.dt == plan.dt for s in mine)
            assert plan.t_start == interval.t_start
            state = door.run.read_flow_state(plan.fingerprint).state
            faces = FaceVelocities.copy_of(np.array(state.u), np.array(state.v))
            transport = _transport(interval, mesh)
            stable = min(
                transport.stable_dt(faces, k)
                for k in range(len(door.config.particle_sizes))
            )
            assert plan.stable_dt == stable
            assert (
                plan.dt
                <= stable
                < (interval.t_end - interval.t_start) / (plan.steps - 1)
            )
        assert n == record.steps
        stables = {p.fingerprint: p.stable_dt for p in record.intervals}
        assert len(set(stables.values())) == 2

    def test_the_hand_over_carries_the_fields_and_budgets_unchanged(
        self, door: SimpleNamespace
    ) -> None:
        """The first step after each switch, stepped by hand from the step before it.

        The fields and budget copies handed to on_step after the last step
        of an interval are given to a new transport solver on the next
        interval's concentration boundary, with that interval's faces, eddy
        viscosity and dt: its step is the run's next step, fields and
        budgets bitwise. Defects caught: the budgets not carried (current
        mass and cumulative terms restart); the fields reset at a switch;
        the old faces or eddy viscosity kept; the old concentration boundary
        kept (the door's corridor air not admitted).
        """
        mesh = Mesh(door.config)
        manager = ScenarioManager(door.config, mesh)
        intervals = manager.intervals()
        by_step = {s.step: s for s in door.steps}
        first = 0
        for plan, interval in zip(door.record.intervals, intervals, strict=True):
            before = by_step.get(first)
            first += plan.steps
            if before is None:
                continue
            after = by_step[before.step + 1]
            state = door.run.read_flow_state(plan.fingerprint).state
            faces = FaceVelocities.copy_of(np.array(state.u), np.array(state.v))
            transport = _transport(interval, mesh)
            transport.restore_budgets(before.budgets)
            for k, c in before.fields.items():
                stepped = transport.solve_timestep(
                    c, faces, k, plan.dt, eddy_viscosity=state.nu_t
                )
                assert stepped.tobytes() == after.fields[k].tobytes(), (plan.index, k)
            same_budgets(tuple(transport.budget), after.budgets)
            for x, y in zip(before.budgets, after.budgets, strict=True):
                assert x.initial == y.initial
        opened = door.steps[door.record.intervals[0].steps]
        assert all(b.inflow > 0.0 for b in opened.budgets)


# ---------------------------------------------------------------------------
# ADR-015 step 3: the arrival maps through the march (REQ-A04)
# ---------------------------------------------------------------------------

# Two sensors in the door's air on 80x30, besides the file's four: one in the
# cell beside the inlet, which is still at or above every threshold when the
# second opening starts, and one a few cells into the room, reached during
# the first opening and below by the second.
DOOR_SENSORS = (
    {"name": "in_door", "x": 0.15, "y": 0.5},
    {"name": "door_room", "x": 0.6, "y": 0.4},
)


def sensed_door_raw(cadence: int | None = 30) -> dict:
    """The door fixture's two openings with DOOR_SENSORS, checkpoints every CADENCE."""
    solver = {} if cadence is None else {"checkpoint_interval": cadence}
    raw = coarse_raw("scenario_door_leak.yaml", **solver)
    raw = rescheduled(raw, 5.0, {"door_open": (1.0, 1.0)})
    raw["scenario"]["events"] = door_events(raw, (3.0, 0.5))
    base = yaml.safe_load(
        (CONFIGS / "clean_room_default.yaml").read_text(encoding="utf-8")
    )
    raw["sensors"] = [*base["sensors"], *map(dict, DOOR_SENSORS)]
    return raw


@pytest.fixture(scope="module")
def sensed_door(
    tmp_path_factory: pytest.TempPathFactory, door: SimpleNamespace
) -> SimpleNamespace:
    """The two-opening door with sensors in its air, every step's fields kept.

    The flow states are the door fixture's, so nothing is solved.
    """
    folder = tmp_path_factory.mktemp("sensed_door")
    config = SimConfig.from_dict(sensed_door_raw())
    steps, on_step = recorder()
    record = run(
        config,
        folder / "run.h5",
        flows_from=door.record.path,
        on_step=on_step,
        provenance=PROVENANCE,
    )
    return SimpleNamespace(
        config=config, record=record, run=RunFile(record.path), steps=steps
    )


def maps_by_rule(
    times: list[float],
    fields: list[Mapping[int, np.ndarray]],
    thresholds: np.ndarray,
    live: np.ndarray,
    onsets: np.ndarray,
) -> np.ndarray:
    """The arrival maps of a series of fields, by section D's rule alone.

    Per event, class and non-SOLID cell, the first of TIMES at or after the
    onset whose field is at or above the threshold; inf where none is, NaN
    in SOLID cells. Written cell by cell from the definition, not from the
    tracker's vectorised update.
    """
    ny, nx = live.shape
    at = np.array(times)
    stack = np.array([[f[k] for k in range(thresholds.size)] for f in fields])
    out = np.full((onsets.size, thresholds.size, ny, nx), np.nan)
    for e, onset in enumerate(onsets):
        for k, threshold in enumerate(thresholds):
            for j, i in zip(*np.nonzero(live), strict=True):
                hits = at[(at >= onset) & (stack[:, k, j, i] >= threshold)]
                out[e, k, j, i] = hits[0] if hits.size else np.inf
    return out


def latencies_by_rule(
    times: np.ndarray, readings: np.ndarray, thresholds: np.ndarray, onsets: np.ndarray
) -> np.ndarray:
    """[events, classes, sensors] from a stored sensor series, by section D's rule."""
    n_classes, n_sensors = readings.shape[1:]
    out = np.full((onsets.size, n_classes, n_sensors), np.inf)
    for e, onset in enumerate(onsets):
        for k in range(n_classes):
            for s in range(n_sensors):
                hits = (times >= onset) & (readings[:, k, s] >= thresholds[k])
                if hits.any():
                    out[e, k, s] = times[np.argmax(hits)] - onset
    return out


@pytest.mark.integration
class TestArrivalMaps:
    def test_the_maps_are_the_rule_on_every_steps_fields(
        self, sensed_door: SimpleNamespace
    ) -> None:
        """/arrival/ is section D's rule applied to every step's fields, bitwise.

        The fields of step 0 (the first frame) and of every step after it
        (on_step's) rebuilt into maps cell by cell. Two events, onsets 1 and
        3 s, each a step time. Defects caught: the last time kept instead of
        the first; ">" for "at or above"; an event's map started at t = 0
        instead of its onset; the tracker not updated at step 0 or after some
        step; the record's unreached cells left NaN.
        """
        record, config = sensed_door.record, sensed_door.config
        arrival = sensed_door.run.read_arrival()
        assert arrival.event_names == ("door_open", "door_open_again")
        assert arrival.onsets.tolist() == [1.0, 3.0]
        assert arrival.t_final == record.t_final == 5.0
        times = [0.0] + [info.t for info in sensed_door.steps]
        assert {1.0, 3.0} <= set(times)
        first = sensed_door.run.read_frames()
        fields = [{k: first.fields[k][0] for k in first.fields}]
        fields += [info.fields for info in sensed_door.steps]
        live = Mesh(config).cell_type != SOLID
        thresholds = np.array(config.class_thresholds())
        expected = maps_by_rule(times, fields, thresholds, live, arrival.onsets)
        assert arrival.first_above.tobytes() == expected.tobytes()
        assert arrival.thresholds.tobytes() == thresholds.tobytes()
        for name in ("thresholds", "onsets", "first_above"):
            assert getattr(record.arrival, name).tobytes() == (
                getattr(arrival, name).tobytes()
            )
        # The first opening reaches cells it the second does not, and the
        # second reaches cells only from its own onset.
        reached = np.isfinite(expected)
        assert reached[0].sum() > reached[1].sum() > 0
        assert (expected[1][reached[1]] >= 3.0).all()

    def test_the_sensors_latencies_from_the_map_equal_the_monitors(
        self, sensed_door: SimpleNamespace
    ) -> None:
        """REQ-A04: the record's two latencies agree to the step, and the stored series' rule.

        in_door is at or above every threshold from the first opening to past
        the second's onset, so the second event reads zero there (an alert
        open at the onset counts as the onset); door_room is reached in the
        first opening only. Defects caught: the two instruments on different
        rules (step 2's window gave inf for in_door's second event); the
        map read at the wrong cell; the latency not taken from the onset.
        """
        record, config = sensed_door.record, sensed_door.config
        assert record.latencies is not None and record.map_latencies is not None
        assert record.latencies.shape == (2, 5, 6)
        assert record.latencies.tobytes() == record.map_latencies.tobytes()
        stored = sensed_door.run.read_sensor_readings()
        expected = latencies_by_rule(
            stored.times,
            stored.readings,
            np.array(config.class_thresholds()),
            record.arrival.onsets,
        )
        assert record.latencies.tobytes() == expected.tobytes()
        in_door, door_room = 4, 5
        assert np.isfinite(record.latencies[0, :, in_door]).all()
        assert (record.latencies[1, :, in_door] == 0.0).all()
        assert (record.latencies[0, :, door_room] > 0.0).all()
        assert np.isinf(record.latencies[1, :, door_room]).all()
        assert np.isinf(record.latencies[:, :, :4]).all()

    def test_the_step_map_against_the_frames_map(
        self, sensed_door: SimpleNamespace
    ) -> None:
        """The frames' map is the step map read at the frames, within one output interval.

        ``arrival_from_frames`` on the run's own frames equals the rule
        applied to the frames' fields alone, bitwise; it is never before the
        step map and reaches no cell the step map does not. Where both reach
        a cell they differ by at most the largest frame spacing on this run
        (a cell can drop below and return later, so that is a measurement,
        not a bound). The prompt expected agreement within one interval at
        every cell; it does not hold at the cells at or above their threshold
        only between two frames, which the frames never see: there are some
        on this run (15 when first measured), each with a step time that is
        no frame's. Defects caught: the frames' map started at t = 0 (cells
        reached before the second onset counted for it); the frames' times
        taken from the wrong axis; a cell the frames miss given the step
        map's time.
        """
        config = sensed_door.config
        frames = sensed_door.run.read_frames()
        arrival = sensed_door.run.read_arrival()
        live = Mesh(config).cell_type != SOLID
        framed = arrival_from_frames(frames, arrival.thresholds, live, arrival.onsets)
        stack = [{k: frames.fields[k][n] for k in frames.fields} for n in range(4)]
        assert frames.steps.size == 4
        expected = maps_by_rule(
            frames.times.tolist(), stack, arrival.thresholds, live, arrival.onsets
        )
        assert framed.tobytes() == expected.tobytes()
        step = arrival.first_above
        both = np.isfinite(step) & np.isfinite(framed)
        assert (framed[both] >= step[both]).all()
        assert not (np.isinf(step) & np.isfinite(framed)).any()
        spacing = float(np.max(np.diff(frames.times)))
        assert (framed[both] - step[both] <= spacing).all()
        missed = np.isfinite(step) & np.isinf(framed)
        assert missed.any()
        assert not np.isin(step[missed], frames.times).any()
        assert np.array_equal(np.isnan(step), np.isnan(framed))

    def test_a_restart_across_a_partly_filled_map_is_the_uninterrupted_run(
        self, tmp_path: Path, door: SimpleNamespace, sensed_door: SimpleNamespace
    ) -> None:
        """Killed after checkpoint 90, between the openings, restarted: /arrival/ bitwise.

        At checkpoint 90 the first opening's map is partly filled and the
        second's empty; the file's /arrival/ is the checkpoint's. Defects
        caught: the tracker's state dropped from the checkpoint (the restart
        refuses for want of it, or a restored empty map reads later times);
        the maps not written at the checkpoint; the restart not rewriting
        them.
        """
        config = sensed_door.config
        path = tmp_path / "run.h5"
        with pytest.raises(_KilledError):
            run(
                config,
                path,
                flows_from=door.record.path,
                on_step=killer(95),
                provenance=PROVENANCE,
            )
        cut = RunFile(path)
        checkpoint = cut.read_checkpoint()
        assert checkpoint.step == 90 and 1.0 < checkpoint.t < 3.0
        partial = cut.read_arrival()
        assert partial.t_final == checkpoint.t
        held = checkpoint.recovery["first_above"]
        assert np.isfinite(held[0]).any() and np.isnan(held[1]).all()
        assert not np.isfinite(held[0][:, Mesh(config).cell_type != SOLID]).all()
        again = run(config, restart=path)
        assert again.restarted_from == 90
        whole = sensed_door.run.read_arrival()
        restarted = cut.read_arrival()
        assert restarted.first_above.tobytes() == whole.first_above.tobytes()
        assert restarted.t_final == whole.t_final
        assert again.latencies is not None
        assert again.latencies.tobytes() == sensed_door.record.latencies.tobytes()

    def test_a_restart_rewrites_the_maps_from_its_checkpoint_at_once(
        self, tmp_path: Path, door: SimpleNamespace, sensed_door: SimpleNamespace
    ) -> None:
        """Maps ahead of the checkpoint are replaced before the march continues.

        A crash between a checkpoint's map write and its checkpoint leaves
        /arrival/ ahead of the checkpoint; planted here by writing the whole
        run's maps after a kill at 95. The restart is killed again at 92,
        before its next checkpoint: the file's maps are checkpoint 90's.
        Defect caught: the restart not rewriting /arrival/ (the whole run's
        maps left in a file whose march stopped at 92).
        """
        config = sensed_door.config
        path = tmp_path / "run.h5"
        with pytest.raises(_KilledError):
            run(
                config,
                path,
                flows_from=door.record.path,
                on_step=killer(95),
                provenance=PROVENANCE,
            )
        cut = RunFile(path)
        checkpoint = cut.read_checkpoint()
        cut.write_arrival(sensed_door.run.read_arrival())
        with pytest.raises(_KilledError):
            run(config, restart=path, on_step=killer(92))
        held = cut.read_arrival()
        assert held.t_final == checkpoint.t
        expected = checkpoint.recovery["first_above"].copy()
        live = np.broadcast_to(Mesh(config).cell_type != SOLID, expected.shape)
        expected[live & np.isnan(expected)] = np.inf
        assert held.first_above.tobytes() == expected.tobytes()

    def test_a_checkpoint_without_the_maps_refuses_the_restart(
        self, tmp_path: Path, door: SimpleNamespace
    ) -> None:
        """A checkpoint from before step 3 carries no maps: KeyError, saying so.

        The three arrays are deleted from a checkpoint the branch wrote.
        """
        config = SimConfig.from_dict(sensed_door_raw())
        path = tmp_path / "run.h5"
        with pytest.raises(_KilledError):
            run(
                config,
                path,
                flows_from=door.record.path,
                on_step=killer(35),
                provenance=PROVENANCE,
            )
        with h5py.File(RunFile(path).checkpoint_path, "a") as checkpoint:
            for name in ARRIVAL_STATE_NAMES:
                del checkpoint["checkpoint/recovery"][name]
        with pytest.raises(KeyError, match=r"before the arrival maps \(ADR-015 step 3"):
            run(config, restart=path)

    def test_a_phase_4_file_is_read_from_its_frames_and_flagged(
        self, tmp_path: Path, sensed_door: SimpleNamespace
    ) -> None:
        """No /arrival/: arrival_from_run rebuilds the maps from the frames, FRAME_RESOLVED.

        The run file is copied and its /arrival/ deleted, as a file written
        before step 3 has none. The onsets come from the configuration the
        file records. With /arrival/ the file's own maps come back,
        STEP_RESOLVED. Defects caught: the frame-resolved maps returned
        unflagged; the onsets taken as t = 0 for a file without the maps; the
        fallback's reason dropped (step 3's open item, prompt 75).
        """
        whole = arrival_from_run(sensed_door.record.path)
        assert whole.resolution == STEP_RESOLVED
        arrival = sensed_door.record.arrival
        assert whole.maps.tobytes() == arrival.first_above.tobytes()
        assert whole.onsets.tobytes() == arrival.onsets.tobytes()
        assert whole.event_names == arrival.event_names
        assert whole.t_final == arrival.t_final and whole.fallback is None
        path = tmp_path / "phase4.h5"
        path.write_bytes(Path(sensed_door.record.path).read_bytes())
        with h5py.File(path, "a") as copy:
            del copy["arrival"]
        old = arrival_from_run(path)
        assert old.resolution == FRAME_RESOLVED
        assert old.onsets.tolist() == [1.0, 3.0]
        assert old.event_names == arrival.event_names
        assert old.fallback == f"KeyError: {path} holds no /arrival/"
        assert old.t_final == RunFile(path).read_frames().times[-1]
        units = detection_units({"old": old, "new": whole})
        n = len(units.labels) // 2
        assert units.frame_resolved == (True,) * n + (False,) * n
        live = Mesh(sensed_door.config).cell_type != SOLID
        expected = arrival_from_frames(
            RunFile(path).read_frames(), arrival.thresholds, live, arrival.onsets
        )
        assert old.maps.tobytes() == expected.tobytes()
        assert arrival_from_run(RunFile(path)).maps.tobytes() == expected.tobytes()

    @pytest.mark.parametrize(
        ("damage", "reason"),
        [
            ("member", r"^KeyError: .*onsets"),
            ("t_final", r"^ValueError: t_final must be finite, got nan$"),
            ("onsets", r"^ValueError: the onsets must be in order"),
        ],
    )
    def test_a_damaged_arrival_group_falls_back_to_the_frames_saying_why(
        self, tmp_path: Path, sensed_door: SimpleNamespace, damage: str, reason: str
    ) -> None:
        """A missing member, a non-finite t_final or unordered onsets: the frames, with the check.

        Step 3's open item (test 74): the fallback hid why. Defect caught:
        the reason not carried (``fallback`` None, or "absent" for a group
        that is there).
        """
        path = tmp_path / "damaged.h5"
        path.write_bytes(Path(sensed_door.record.path).read_bytes())
        with h5py.File(path, "a") as copy:
            group = copy["arrival"]
            if damage == "member":
                del group["onsets"]
            elif damage == "t_final":
                group.attrs["t_final"] = np.nan
            else:
                group["onsets"][...] = group["onsets"][()][::-1]
        old = arrival_from_run(path)
        assert old.resolution == FRAME_RESOLVED
        assert re.search(reason, old.fallback), old.fallback
        assert "holds no /arrival/" not in old.fallback

    def test_a_clean_out_read_from_its_frames_reads_zero_at_every_live_cell(
        self, tmp_path: Path, clean_out: SimpleNamespace
    ) -> None:
        """No /arrival/: the clean-out's first frame is at its onset, so every cell reads 0.0.

        The one event starts at t = 0 and the first frame is at t = 0, where
        every non-SOLID cell is at or above every threshold; the frames' map
        agrees with the file's own /arrival/ there. Defect caught: ">" for
        ">=" in arrival_from_frames' comparison with the onset (every cell
        reading the second frame's time). The held run stops before its
        t_end, and the frames' t_final is its last frame's, as /arrival/'s is
        its last step's. Defect caught: the frames' t_final taken from the
        configuration's t_end (prompt 75).
        """
        path = tmp_path / "clean_out.h5"
        path.write_bytes(Path(clean_out.record.path).read_bytes())
        with h5py.File(path, "a") as copy:
            del copy["arrival"]
        old = arrival_from_run(path)
        assert old.resolution == FRAME_RESOLVED
        assert old.onsets.tolist() == [0.0]
        assert RunFile(path).read_frames().times[0] == 0.0
        live = Mesh(clean_out.config).cell_type != SOLID
        assert (old.maps[:, :, live] == 0.0).all()
        assert np.isnan(old.maps[:, :, ~live]).all()
        whole = arrival_from_run(clean_out.record.path)
        assert whole.resolution == STEP_RESOLVED
        assert old.maps.tobytes() == whole.maps.tobytes()
        assert old.t_final == whole.t_final < clean_out.config.t_end

    def test_every_run_has_maps_and_no_sensors_no_latencies(
        self, tmp_path: Path, clean_out: SimpleNamespace
    ) -> None:
        """A run without sensors still writes /arrival/; its record has no latencies.

        The clean-out has one event, its initial, at t = 0: every non-SOLID
        cell is at or above every threshold then, so its map is t = 0
        everywhere (section D: the clearing companion says nothing new).
        """
        arrival = clean_out.run.read_arrival()
        assert arrival.event_names == ("contaminated_start",)
        live = Mesh(clean_out.config).cell_type != SOLID
        assert (arrival.first_above[:, :, live] == 0.0).all()
        assert clean_out.record.latencies is not None
        assert (clean_out.record.latencies == 0.0).all()
        raw = rescheduled(
            coarse_raw("scenario_equipment_dust.yaml", output_interval=10),
            1.0,
            {"litho_dust": (0.5, 0.25)},
        )
        raw["sensors"] = []
        record = run(
            SimConfig.from_dict(raw),
            tmp_path / "run.h5",
            flows_from=clean_out.record.path,
            provenance=PROVENANCE,
        )
        assert record.latencies is None and record.map_latencies is None
        assert RunFile(record.path).read_arrival().event_names == ("litho_dust",)
        assert run_scenario.latency_lines(record, [1e-7] * 5, []) == [
            "latency (REQ-A04): no sensors"
        ]

    def test_a_run_without_a_class_threshold_is_refused_for_its_maps(
        self, tmp_path: Path
    ) -> None:
        """Without sensors there is no monitor, but every run keeps the maps.

        Defect caught: a run without sensors starting with no map.
        """
        raw = rescheduled(
            coarse_raw("scenario_equipment_dust.yaml"), 2.0, {"litho_dust": (0.5, 1.0)}
        )
        raw["particles"] = {"sizes": [0.1e-6, 0.3e-6, 0.5e-6, 1.0e-6, 2.0e-6]}
        raw["sensors"] = []
        with pytest.raises(ValueError, match="the arrival maps: the class of diam"):
            run(SimConfig.from_dict(raw), tmp_path / "run.h5", provenance=PROVENANCE)


def _transport(interval: Interval, mesh: Mesh) -> TransportSolver:
    """A transport solver on an interval's boundary, built without the run."""
    config = interval.config
    physics = ParticlePhysics(config)
    boundary = ConcentrationBoundary(mesh, config, physics, BoundaryRegistry(config))
    return TransportSolver(mesh, config, physics, boundary)


# ---------------------------------------------------------------------------
# The restart
# ---------------------------------------------------------------------------


def _door_clean_out(cadence: int | None) -> SimConfig:
    """The clean-out with the door open from 1 s to 2 s, to 4 s: a switch and a tracker."""
    raw = coarse_raw("scenario_clean_out.yaml")
    door_raw = coarse_raw("scenario_door_leak.yaml")
    opening = door_raw["scenario"]["events"][0]
    opening["start"], opening["duration"] = 1.0, 1.0
    raw["scenario"]["events"].append(opening)
    raw["solver"]["t_end"] = 4.0
    if cadence is not None:
        raw["solver"]["checkpoint_interval"] = cadence
    return SimConfig.from_dict(raw)


def _with_t_end(config: SimConfig, t_end: float) -> SimConfig:
    """CONFIG with solver.t_end replaced, a key a restart may change."""
    raw = config.merged_mapping()
    raw["solver"]["t_end"] = t_end
    return SimConfig.from_dict(raw)


def _copy_run(source: Path, target: Path) -> None:
    """A run file and its checkpoint file copied byte for byte."""
    target.write_bytes(source.read_bytes())
    copy = RunFile(target).checkpoint_path
    copy.write_bytes(RunFile(source).checkpoint_path.read_bytes())


@pytest.mark.integration
class TestRestart:
    def test_after_a_switch_the_restart_is_the_uninterrupted_run(
        self, tmp_path: Path, door: SimpleNamespace
    ) -> None:
        """Killed between two checkpoints after the door opened, restarted, finished.

        The fields, frames, budgets, sensor readings and recovery outputs,
        the local field included, are the uninterrupted run's bitwise.
        Checkpoints every 30 steps, frames every 100; checkpoint 90 follows
        the switch. Killed at 95, before frame 100: the readings since frame
        0 are in the file only because they were appended before the
        checkpoint. Killed at 107, after frame 100: frame 100 and its
        readings must be dropped before the march writes them again.
        Defects caught: the readings held for the next frame not flushed
        before a checkpoint (a gap of 90 steps, the kill at 95); the frame
        and readings written after the checkpoint kept (frame 100 refused
        on the way back, the kill at 107); the tracker's state not carried;
        the restart resuming in the first interval's plan; the budgets not
        restored.
        """
        config = _door_clean_out(30)
        kept = door.record.path
        whole = run(
            config, tmp_path / "whole.h5", flows_from=kept, provenance=PROVENANCE
        )
        switch = whole.intervals[0].steps
        frame, checkpoint = 100, 90
        assert switch < checkpoint < frame < whole.steps
        for kill, frames in ((95, [0]), (frame + 7, [0, frame])):
            cut = tmp_path / f"cut_{kill}.h5"
            with pytest.raises(_KilledError):
                run(
                    config,
                    cut,
                    flows_from=kept,
                    on_step=killer(kill),
                    provenance=PROVENANCE,
                )
            assert RunFile(cut).read_checkpoint().step == checkpoint
            assert RunFile(cut).read_frames().steps.tolist() == frames
            again = run(config, restart=cut)
            assert again.restarted_from == checkpoint
            assert [f.origin for f in again.flows] == [RUN_FILE, RUN_FILE]
            assert (again.steps, again.t_final) == (whole.steps, whole.t_final)
            same_frames(RunFile(cut), RunFile(whole.path))
            same_budgets(again.budgets, whole.budgets)
            same_recovery(again.recovery, whole.recovery)
            same_recovery(RunFile(cut).read_recovery(), whole.recovery)

    def test_without_a_checkpoint_the_march_runs_again(
        self, tmp_path: Path, door: SimpleNamespace
    ) -> None:
        """No checkpoint_interval: the restart marches from step 0 on the kept states.

        Defects caught: the frames and readings of the killed march kept, so
        frame 0 is refused or written twice.
        """
        config = _door_clean_out(None)
        kept = door.record.path
        whole = run(
            config, tmp_path / "whole.h5", flows_from=kept, provenance=PROVENANCE
        )
        cut = tmp_path / "cut.h5"
        with pytest.raises(_KilledError):
            run(config, cut, flows_from=kept, on_step=killer(80), provenance=PROVENANCE)
        assert not RunFile(cut).checkpoint_path.exists()
        again = run(config, restart=cut)
        assert again.restarted_from == 0
        same_frames(RunFile(cut), RunFile(whole.path))
        same_recovery(again.recovery, whole.recovery)

    def test_a_restart_without_a_clean_out_restores_no_tracker(
        self, tmp_path: Path, door: SimpleNamespace
    ) -> None:
        """The door leak has no recovery tracker, and its restart is bitwise too.

        Its checkpoint carries no recovery arrays and the restart must not
        look for any. Killed at 95, after the checkpoint at 90 that follows
        the first opening. Defects caught: a tracker restored from a
        checkpoint that has none (an error on the way back); a restart that
        marches from step 0; the frames after the checkpoint kept.
        """
        cut = tmp_path / "cut.h5"
        with pytest.raises(_KilledError):
            run(
                door.config,
                cut,
                flows_from=door.record.path,
                on_step=killer(95),
                provenance=PROVENANCE,
            )
        assert RunFile(cut).read_checkpoint().step == 90
        again = run(door.config, restart=cut)
        assert again.restarted_from == 90
        assert again.recovery is None
        same_frames(RunFile(cut), door.run)
        same_budgets(again.budgets, door.record.budgets)

    @pytest.mark.parametrize("written", ["without", "other_value"])
    def test_a_file_written_without_or_under_other_placement_restarts(
        self, tmp_path: Path, door: SimpleNamespace, written: str
    ) -> None:
        """A run file made before the placement section, or under other values, restarts.

        The door run is killed at 95 under a base file that states no
        placement section (a Phase 4 file's configuration) or another
        count and rule, and restarted under the same base with the product
        file's section stated in the scenario file: accepted, from
        checkpoint 90, and the frames, budgets and readings are the
        uninterrupted run's bitwise (the section changes nothing the march
        computes). The flow fingerprint is the same with and without the
        section. Defects caught: the section left in the restart hash (the
        restart refused naming ``placement``); the section read by the flow
        fingerprint (the kept state not matched).
        """
        base = yaml.safe_load(
            (CONFIGS / "clean_room_default.yaml").read_text(encoding="utf-8")
        )
        if written == "without":
            del base["placement"]
        else:
            base["placement"].update({"count": 2, "rule": "mean"})
        base_path = tmp_path / "base.yaml"
        base_path.write_text(yaml.safe_dump(base, sort_keys=False), encoding="utf-8")
        raw = yaml.safe_load(yaml.safe_dump(door.raw))
        raw["scenario"]["base"] = str(base_path)
        older = SimConfig.from_dict(raw)
        assert ("placement" in older.merged_mapping()) == (written != "without")
        cut = tmp_path / "cut.h5"
        with pytest.raises(_KilledError):
            run(
                older,
                cut,
                flows_from=door.record.path,
                on_step=killer(95),
                provenance=PROVENANCE,
            )
        raw["placement"] = yaml.safe_load(
            (CONFIGS / "clean_room_default.yaml").read_text(encoding="utf-8")
        )["placement"]
        newer = SimConfig.from_dict(raw)
        assert newer.placement == door.config.placement
        assert flow_fingerprint(newer) == flow_fingerprint(older)
        again = run(newer, restart=cut)
        assert again.restarted_from == 90
        same_frames(RunFile(cut), door.run)
        same_budgets(again.budgets, door.record.budgets)

    def test_a_later_t_end_divides_the_last_interval_again(
        self, tmp_path: Path, door: SimpleNamespace
    ) -> None:
        """A finished run restarted under a later t_end continues from its end.

        Defects caught: the old plan's step index reused under the new
        interval's dt (the time jumps); the steps after the restart not
        appended after the old ones.
        """
        config = _door_clean_out(30)
        first = run(
            config,
            tmp_path / "run.h5",
            flows_from=door.record.path,
            provenance=PROVENANCE,
        )
        later = _with_t_end(config, 4.5)
        steps, on_step = recorder()
        again = run(later, restart=tmp_path / "run.h5", on_step=on_step)
        assert again.restarted_from == first.steps
        plan = again.resumed_plan
        assert plan is not None
        assert (plan.t_start, plan.t_end) == (first.t_final, 4.5)
        assert [s.t for s in steps] == [
            first.t_final + i * plan.dt for i in range(1, plan.steps)
        ] + [4.5]
        assert again.t_final == 4.5 and again.steps == first.steps + plan.steps
        frames = RunFile(tmp_path / "run.h5").read_frames()
        assert frames.steps[-1] == again.steps
        assert np.all(np.diff(frames.steps) > 0)

    def test_a_second_restart_continues_on_the_extensions_plan(
        self, tmp_path: Path, door: SimpleNamespace
    ) -> None:
        """An extended run killed after a checkpoint, restarted again, finished.

        A run to 4 s is extended to 4.5 s twice from copies of its files:
        once uninterrupted, once killed three steps after the first
        checkpoint of the extension (every 10 steps) and restarted under the
        same 4.5 s. The checkpoint lies on the plan the extension divided
        again from 4 s, not on the plan
        from the interval's start, and the second restart continues on it:
        frames, budgets, readings and recovery outputs bitwise. Defects
        caught: the plan's start not written with the checkpoint, or not
        read by the restart, so the rest is divided again from the
        checkpoint's time with another step (review 64 S4.1).
        """
        config = _door_clean_out(10)
        first = run(
            config,
            tmp_path / "whole.h5",
            flows_from=door.record.path,
            provenance=PROVENANCE,
        )
        _copy_run(tmp_path / "whole.h5", tmp_path / "cut.h5")
        later = _with_t_end(config, 4.5)
        whole = run(later, restart=tmp_path / "whole.h5")
        assert whole.resumed_plan is not None
        assert whole.resumed_plan.t_start == first.t_final
        checkpoint = 10 * (first.steps // 10 + 1)
        assert first.steps < checkpoint < checkpoint + 3 < whole.steps
        with pytest.raises(_KilledError):
            run(later, restart=tmp_path / "cut.h5", on_step=killer(checkpoint + 3))
        again = run(later, restart=tmp_path / "cut.h5")
        assert again.restarted_from == checkpoint
        assert again.resumed_plan == whole.resumed_plan
        assert (again.steps, again.t_final) == (whole.steps, 4.5)
        same_frames(RunFile(tmp_path / "cut.h5"), RunFile(whole.path))
        same_budgets(again.budgets, whole.budgets)
        same_recovery(again.recovery, whole.recovery)

    @pytest.mark.parametrize("t_end", [4.0, 3.5])
    def test_a_t_end_at_or_before_the_checkpoint_is_refused(
        self, tmp_path: Path, door: SimpleNamespace, t_end: float
    ) -> None:
        """A finished run at 4 s restarted under 4 s or 3.5 s: refused, naming both.

        The run file is left as it was. Defects caught: the restart accepted
        and the run reported past its t_end (review 64 S4.2); the refusal
        strict, so a restart under the checkpoint's own time is accepted.
        """
        config = _door_clean_out(30)
        path = tmp_path / "run.h5"
        first = run(config, path, flows_from=door.record.path, provenance=PROVENANCE)
        assert first.t_final == 4.0
        before = RunFile(path).read_frames().steps.tolist()
        message = (
            rf"restart refused: the checkpoint is at t = 4\.0 s \(step "
            rf"{first.steps}\), at or past solver\.t_end \({t_end!r} s\)"
        )
        with pytest.raises(ValueError, match=message):
            run(_with_t_end(config, t_end), restart=path)
        assert RunFile(path).read_frames().steps.tolist() == before

    def test_a_checkpoint_on_an_intervals_last_step(
        self, tmp_path: Path, door: SimpleNamespace
    ) -> None:
        """The cadence is the first interval's step count: a checkpoint at its end.

        The checkpoint's time is the interval's end exactly, which is the
        next interval's start, so the restart continues at the next plan's
        step 0. Killed two steps after it and restarted: the uninterrupted
        run bitwise. Defect caught: the checkpoint taken as the end of its
        own interval, which the plan has no step after (review 64 S4.4).
        """
        kept = door.record.path
        whole = run(
            _door_clean_out(None),
            tmp_path / "whole.h5",
            flows_from=kept,
            provenance=PROVENANCE,
        )
        first = whole.intervals[0]
        config = _door_clean_out(first.steps)
        cut = tmp_path / "cut.h5"
        with pytest.raises(_KilledError):
            run(
                config,
                cut,
                flows_from=kept,
                on_step=killer(first.steps + 2),
                provenance=PROVENANCE,
            )
        checkpoint = RunFile(cut).read_checkpoint()
        assert (checkpoint.step, checkpoint.t) == (first.steps, first.t_end)
        assert checkpoint.fingerprint == first.fingerprint
        again = run(config, restart=cut)
        assert again.restarted_from == first.steps
        assert again.resumed_plan is None
        assert (again.steps, again.t_final) == (whole.steps, whole.t_final)
        same_frames(RunFile(cut), RunFile(whole.path))
        same_budgets(again.budgets, whole.budgets)
        same_recovery(again.recovery, whole.recovery)

    def test_a_diverged_flow_state_stops_the_run(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A solve that leaves no flow_state (non-finite fields) raises, writing nothing.

        The solver is replaced by one that drops its flow state after five
        outer iterations, as a diverged solve leaves it. Defect caught: a
        missing state passed on, failing later with no flow state named.
        """

        class Diverged(time_integration.StaggeredSolver):
            def solve_steady(self, *args: object, **kwargs: object) -> object:
                out = super().solve_steady(*args, **kwargs)
                self.flow_state = self.steady_checkpoint = None
                return out

        monkeypatch.setattr(time_integration, "StaggeredSolver", Diverged)
        raw = coarse_raw("scenario_clean_out.yaml", max_simple_iter=5)
        raw = rescheduled(raw, 6.0, {})
        path = tmp_path / "run.h5"
        with pytest.raises(FlowNotConvergedError, match="after 5 outer"):
            run(SimConfig.from_dict(raw), path, provenance=PROVENANCE)
        assert RunFile(path).flow_fingerprints() == ()
        assert not RunFile(path).checkpoint_path.exists()

    def test_a_capped_flow_solve_continues_under_a_larger_cap(
        self, tmp_path: Path, clean_out: SimpleNamespace
    ) -> None:
        """Capped at 300 outer iterations, then restarted with the file's cap.

        The capped state leaves its steady checkpoint and no flow state; the
        restart resumes it to the uninterrupted solve's state bitwise.
        Defects caught: a capped state written to /flow/ and marched on; the
        restart solving from rest (the outer count and faces differ only in
        the stop's iteration error, and the origin says so).
        """
        raw = rescheduled(coarse_raw("scenario_clean_out.yaml"), 6.0, {})
        capped = SimConfig.from_dict(
            {**raw, "solver": {**raw["solver"], "max_simple_iter": 300}}
        )
        path = tmp_path / "run.h5"
        with pytest.raises(FlowNotConvergedError, match="max_simple_iter after 300"):
            run(capped, path, provenance=PROVENANCE)
        assert RunFile(path).flow_fingerprints() == ()
        key, steady = RunFile(path).read_steady_checkpoint()
        assert key == clean_out.record.flows[0].fingerprint
        assert steady.state.outer == 300
        again = run(SimConfig.from_dict(raw), restart=path)
        assert [(f.origin, f.outer) for f in again.flows] == [(RESUMED, COARSE_OUTER)]
        state = RunFile(path).read_flow_state(key).state
        theirs = clean_out.run.read_flow_state(key).state
        same_faces(state, theirs)
        for field in ("p", "k", "eps", "nu_t"):
            assert getattr(state, field).tobytes() == getattr(theirs, field).tobytes()


# ---------------------------------------------------------------------------
# scripts/run_scenario.py
# ---------------------------------------------------------------------------


def _scenario_file(tmp_path: Path, raw: dict) -> Path:
    path = tmp_path / "scenario.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


@pytest.mark.integration
class TestRunScenario:
    def test_a_coarse_clean_out_and_its_summary(
        self,
        tmp_path: Path,
        clean_out: SimpleNamespace,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Exit 0 and the summary's lines: intervals, states, steps, budgets, recovery.

        Defects caught: a summary line dropped; the reuse counted wrong; the
        exit status not 0 on a finished run.
        """
        raw = rescheduled(coarse_raw("scenario_clean_out.yaml"), 6.0, {})
        scenario = _scenario_file(tmp_path, raw)
        out = tmp_path / "run.h5"
        argv = [str(scenario), "--out", str(out), "--flows-from"]
        argv += [str(clean_out.record.path), "--quiet"]
        assert run_scenario.main(argv) == run_scenario.EXIT_DONE
        lines = capsys.readouterr().out.splitlines()
        record = RunFile(out)
        assert lines[0] == f"scenario: {scenario}"
        assert lines[1] == f"run file: {out}"
        assert lines[2] == "intervals: 1"
        assert lines[3].startswith("  [0, 6) s: state ")
        assert lines[4] == (
            "flow states: 1 distinct, 0 solved, 1 read from a run file; intervals on "
            "a state already held: 0"
        )
        assert lines[5].endswith(
            f"kept, {COARSE_OUTER} outer, error_estimate_and_continuity, "
            f"{clean_out.record.flows[0].wall_seconds:.1f} s"
        )
        frames = record.read_frames()
        assert lines[6] == f"steps: {frames.steps[-1]} to t = 6 s (t_end)"
        assert lines[7].startswith("wall time: ")
        assert lines[8].startswith("budget closure")
        assert [line.split(":")[0] for line in lines[9:14]] == [
            f"  class {k} ({d * 1e6:g} um)"
            for k, d in enumerate(clean_out.config.particle_sizes)
        ]
        assert lines[14] == "recovery (ADR-014 C):"
        assert "  room recovery time: not recovered by 6.0000 s" in lines
        assert (
            sum("free-area mean 100:1: not reached by 6.0000 s" in x for x in lines)
            == 5
        )
        # The initial field is 100 thresholds everywhere, so every sensor's
        # class opens at step 0 (ADR-015 C).
        table = record.read_alerts()
        start = lines.index(
            f"alerts (ADR-015 C): {len(table)} opened, "
            f"{int(np.sum(table.step_off == -1))} still open at the end"
        )
        assert len(table) == 20 and (table.step_on == 0).all()
        latency = start + 1 + len(table)
        named = [line.split(",")[0].strip() for line in lines[start + 1 : latency]]
        assert named == [table.names[s] for s in table.sensor]
        # Every sensor is at or above at the initial event's onset, t = 0.
        assert lines[latency] == (
            "latency (REQ-A04, ADR-015 D): from each event's onset, the monitor's "
            "and the arrival map's at the sensor's cell, which agree"
        )
        assert lines[latency + 1] == "  event contaminated_start, onset t = 0.0000 s:"
        assert len(lines) == latency + 2 + 5
        for line in lines[latency + 2 :]:
            assert line.count("0.0000 s, map 0.0000 s") == 4, line

    def test_a_class_nothing_supplied_is_not_defined_in_the_summary(
        self,
        tmp_path: Path,
        clean_out: SimpleNamespace,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """A release of one class: the other four classes' closure is not defined.

        The supply is clean and there is no initial event, so classes 0, 1,
        3 and 4 are never supplied and their relative closure has nothing to
        divide by. Defect caught: the summary calls MassBudget.relative()
        for them and the finished run ends in a traceback with exit 1
        (review 64 S2).
        """
        raw = rescheduled(
            coarse_raw("scenario_equipment_dust.yaml"),
            2.0,
            {"litho_dust": (0.5, 1.0)},
        )
        raw["scenario"]["events"][0]["source"]["emission"] = [0, 0, 1.0e4, 0, 0]
        scenario = _scenario_file(tmp_path, raw)
        argv = [str(scenario), "--out", str(tmp_path / "run.h5"), "--quiet"]
        argv += ["--flows-from", str(clean_out.record.path)]
        assert run_scenario.main(argv) == run_scenario.EXIT_DONE
        lines = capsys.readouterr().out.splitlines()
        start = lines.index(
            "budget closure (residual, relative to the particles supplied; the "
            "implicit solve's leftover, booked, ADR-016):"
        )
        closure = lines[start + 1 : start + 6]
        for k, line in enumerate(closure):
            assert line.startswith(f"  class {k} (")
            supplied = "relative not defined (nothing supplied)" in line
            assert supplied == (k != 2), line
        assert "relative " in closure[2]
        assert len(RunFile(tmp_path / "run.h5").read_alerts()) == 0
        assert lines[-8] == (
            "alerts (ADR-015 C): none (no sensor at or above its threshold)"
        )
        assert lines[-6] == "  event litho_dust, onset t = 0.5000 s:"
        for line in lines[-5:]:
            assert line.count("never by 2.0000 s, map never") == 4, line

    def test_the_alert_lines_name_each_alert_and_its_close(self) -> None:
        """alert_lines on planted alerts: the count, an open and a closed line.

        Defects caught: an open alert printed with a closing time; the class
        named by index into the wrong list; a run without sensors reported
        as one with no alerts.
        """
        closed = Alert("hood_entry", 1, 3, 0.3, 2.0e4, 5, 0.5, 3.0e4, 0.4)
        still = Alert("near_door", 4, 7, 0.7, 30.0, None, None, 31.5, 0.8)
        lines = run_scenario.alert_lines(
            [closed, still], [1e-7, 3e-7, 0, 0, 5e-6], True
        )
        assert lines == [
            "alerts (ADR-015 C): 2 opened, 1 still open at the end",
            "  hood_entry, class 1 (0.3 um): at or above from t = 0.3000 s (step 3), "
            "reading 2e+04; below at t = 0.5000 s (step 5); peak 3e+04 at t = 0.4000 s",
            "  near_door, class 4 (5 um): at or above from t = 0.7000 s (step 7), "
            "reading 30; still open; peak 31.5 at t = 0.8000 s",
        ]
        assert run_scenario.alert_lines([], [1e-7], False) == [
            "alerts (ADR-015 C): no sensors"
        ]

    def test_the_latency_lines_give_both_instruments_and_never(self) -> None:
        """latency_lines on a planted record: a time, a never and a disagreement.

        Defects caught: the map's column printed from the monitor's array; a
        disagreement not marked; "never" printed without the run's end; the
        events or classes misnamed.
        """
        monitor = np.array([[[0.25, np.inf]], [[0.0, 1.5]]])
        mapped = monitor.copy()
        mapped[1, 0, 1] = 1.75
        arrival = SimpleNamespace(event_names=("a", "b"), onsets=np.array([1.0, 3.0]))
        record = SimpleNamespace(
            latencies=monitor, map_latencies=mapped, arrival=arrival, t_final=6.0
        )
        lines = run_scenario.latency_lines(record, [2e-6], ["s0", "s1"])  # type: ignore[arg-type]
        assert lines == [
            "latency (REQ-A04, ADR-015 D): from each event's onset, the monitor's and "
            "the arrival map's at the sensor's cell, which DISAGREE",
            "  event a, onset t = 1.0000 s:",
            "    class 0 (2 um): s0 0.2500 s, map 0.2500 s; s1 never by 6.0000 s, map "
            "never",
            "  event b, onset t = 3.0000 s:",
            "    class 0 (2 um): s0 0.0000 s, map 0.0000 s; s1 1.5000 s, map 1.7500 s "
            "DISAGREES",
        ]

    def test_the_restart_flag_continues_a_run_file(
        self,
        tmp_path: Path,
        clean_out: SimpleNamespace,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """--restart on a copy of the held clean-out: the restart and hold lines.

        Defects caught: the flag not passed to run (the script refuses or
        starts a new run); the restart line or the hold's stop line dropped
        (test 64 S1).
        """
        raw = held_clean_out(checkpoint_interval=500)
        scenario = _scenario_file(tmp_path, raw)
        path = tmp_path / "run.h5"
        _copy_run(clean_out.record.path, path)
        argv = [str(scenario), "--restart", str(path), "--quiet"]
        assert run_scenario.main(argv) == run_scenario.EXIT_DONE
        lines = capsys.readouterr().out.splitlines()
        record = clean_out.record
        assert lines[2] == f"restarted from step {record.steps}"
        assert (
            f"steps: {record.steps} to t = {record.t_final:.6g} s (the hold)" in lines
        )
        assert f"  stopped by the hold of {ONE_AIR_CHANGE} s" in lines
        assert lines[4].startswith("  [0, 120) s: state ")

    def test_the_exit_codes_of_a_refusal_and_a_capped_solve(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """2 for an existing run file, 1 for a flow state stopped at its cap.

        Defects caught: a refusal or a capped solve reported as a finished run.
        """
        raw = rescheduled(
            coarse_raw("scenario_clean_out.yaml", max_simple_iter=5), 6.0, {}
        )
        scenario = _scenario_file(tmp_path, raw)
        out = tmp_path / "run.h5"
        argv = [str(scenario), "--out", str(out), "--quiet"]
        assert run_scenario.main(argv) == run_scenario.EXIT_NOT_CONVERGED
        assert "max_simple_iter after 5 outer" in capsys.readouterr().err
        assert run_scenario.main(argv) == run_scenario.EXIT_REFUSED
        assert "exists; a new run writes new files" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# ADR-016: the conservation budget on the 80x30 room (prompt 78)
# ---------------------------------------------------------------------------

# The dust release's window, rescheduled as TRANSPORT_ONLY's, at a tolerance
# whose max-norm stop alone leaves more than the share on most release steps,
# so the share is what holds the leftover there.
BUDGET_WINDOW_TOL = 1.0e-5
BUDGET_WINDOW = ("scenario_equipment_dust.yaml", 4.0, {"litho_dust": (1.0, 2.0)})


def budget_window_raw(**transport: object) -> dict:
    """The coarse dust release from 1 s to 3 s, to 4 s, with TRANSPORT's keys set."""
    name, t_end, times = BUDGET_WINDOW
    raw = rescheduled(coarse_raw(name, output_interval=10), t_end, times)
    raw.setdefault("transport", {})["diffusion_tol"] = BUDGET_WINDOW_TOL
    raw["transport"].update(transport)
    return raw


@pytest.fixture(scope="module")
def budget_window(
    tmp_path_factory: pytest.TempPathFactory, clean_out: SimpleNamespace
) -> SimpleNamespace:
    """The window on the clean-out's kept state, every step recorded."""
    folder = tmp_path_factory.mktemp("budget_window")
    raw = budget_window_raw()
    raw["solver"]["checkpoint_interval"] = 50
    config = SimConfig.from_dict(raw)
    steps, on_step = recorder()
    record = run(
        config,
        folder / "run.h5",
        flows_from=clean_out.record.path,
        on_step=on_step,
        provenance=PROVENANCE,
    )
    return SimpleNamespace(config=config, record=record, steps=steps)


def _strip_leftover(path: Path) -> None:
    """Rewrite a run file's and its checkpoint's budgets as a writer before prompt 78 did.

    The leftover column is dropped and the terms attribute is the nine terms
    without it, which is what every run file written before the term was
    booked holds.
    """
    targets = ((path, TRANSPORT_GROUP), (RunFile(path).checkpoint_path, "checkpoint"))
    for target, group in targets:
        with h5py.File(target, "a") as handle:
            budgets = handle[group]["budgets"]
            terms = [str(t) for t in budgets.attrs["terms"]]
            assert terms[-1] == "leftover"
            rows = budgets[()][..., :-1]
            del handle[group]["budgets"]
            if group == TRANSPORT_GROUP:
                stored = handle[group].create_dataset(
                    "budgets",
                    data=rows,
                    maxshape=(None, *rows.shape[1:]),
                    chunks=(1, *rows.shape[1:]),
                )
            else:
                stored = handle[group].create_dataset("budgets", data=rows)
            stored.attrs["terms"] = terms[:-1]


TRANSPORT_GROUP = "transport"


@pytest.mark.integration
class TestConservationBudget:
    def test_the_window_closes_to_rounding_at_every_step_and_class(
        self, budget_window: SimpleNamespace
    ) -> None:
        """With the leftover booked, every step's budget closes within 1e-12 of the supply.

        Across the release's onset and end, every class. Without the term the
        residual is the leftover itself, which here exceeds 1e-8 of the
        supply. Defect caught: the booking removed (budget.leftover not added).
        """
        steps = budget_window.steps
        assert len(steps) == budget_window.record.steps
        largest_leftover = 0.0
        for info in steps:
            for budget in info.budgets:
                supplied = budget.supplied()
                if supplied == 0.0:
                    assert budget.residual() == 0.0 and budget.leftover == 0.0
                    continue
                assert abs(budget.residual()) <= 1e-12 * supplied
                largest_leftover = max(
                    largest_leftover, abs(budget.leftover_relative())
                )
        assert largest_leftover > 1e-8

    def test_every_steps_leftover_is_within_its_share_and_the_bound_holds(
        self, budget_window: SimpleNamespace
    ) -> None:
        """|leftover_n| <= f L S_n dt_n / T at every step; the sum within f L S(t) t / T.

        S_n is the supply after the step, the step's inflow and source booked
        before its solve. A relative allowance of 1e-9 covers the difference
        of two accumulated sums. Defect caught: the share ignored in the stop
        (the max-norm stop alone at BUDGET_WINDOW_TOL leaves the release's
        leftover above its share).
        """
        config = budget_window.config
        limit = config.transport.conservation_limit
        share = config.transport.leftover_share
        horizon = config.t_end
        previous = [0.0] * len(config.particle_sizes)
        supplied_steps = 0
        for info in budget_window.steps:
            for k, budget in enumerate(info.budgets):
                supplied = budget.supplied()
                step = budget.leftover - previous[k]
                previous[k] = budget.leftover
                allowed = share * limit * supplied * info.dt / horizon
                assert abs(step) <= allowed * (1.0 + 1e-9), (info.step, k)
                bound = share * limit * supplied * info.t / horizon
                assert abs(budget.leftover) <= bound * (1.0 + 1e-9), (info.step, k)
                supplied_steps += supplied > 0.0
        assert supplied_steps > 0

    def test_the_share_bites_in_the_window(
        self, budget_window: SimpleNamespace
    ) -> None:
        """Somewhere in the window a step's leftover is above a tenth of its share.

        Leftovers far below the share at every step would leave the test above
        unable to fail; at BUDGET_WINDOW_TOL the max-norm stop leaves more, so
        the share's own sweeps end some steps just inside it.
        """
        config = budget_window.config
        f_l = config.transport.leftover_share * config.transport.conservation_limit
        previous = [0.0] * len(config.particle_sizes)
        close = 0
        for info in budget_window.steps:
            for k, budget in enumerate(info.budgets):
                step = budget.leftover - previous[k]
                previous[k] = budget.leftover
                allowed = f_l * budget.supplied() * info.dt / config.t_end
                close += allowed > 0.0 and abs(step) > 0.1 * allowed
        assert close > 0

    def test_a_breach_exits_1_and_leaves_the_checkpoint(
        self,
        tmp_path: Path,
        clean_out: SimpleNamespace,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """One sweep per solve: the share cannot hold and the run is refused.

        scripts/run_scenario.py reports the class and the limit and exits 1;
        the checkpoint written before the breach is left. Defect caught: the
        refusal's check removed (the run then finishes and exits 0).
        """
        raw = budget_window_raw(max_diffusion_iter=1)
        raw["solver"]["checkpoint_interval"] = 20
        scenario = _scenario_file(tmp_path, raw)
        out = tmp_path / "run.h5"
        argv = [
            str(scenario), "--out", str(out), "--flows-from",
            str(clean_out.record.path), "--quiet",
        ]  # fmt: skip
        assert run_scenario.main(argv) == run_scenario.EXIT_NOT_CONVERGED
        err = capsys.readouterr().err
        assert "accumulated leftover" in err and "conservation_limit" in err
        checkpoint = RunFile(out).read_checkpoint()
        assert checkpoint.step > 0
        assert all(b.leftover is not None for b in checkpoint.budgets)
        assert checkpoint.t < BUDGET_WINDOW[1]

    def test_an_older_run_file_reads_and_its_restart_is_refused(
        self,
        tmp_path: Path,
        budget_window: SimpleNamespace,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """The window's files rewritten as before prompt 78: read, leftover unknown.

        The frames read with the leftover None and the residual they always
        had, the booked residual plus the leftover; a restart is refused
        before anything is written, naming the unknown leftover, and the
        runner exits 2; no frame is appended to the older rows. Defects
        caught: the reader zipping the rows with the current terms (an older
        file then fails to read); the restart's check removed (the refusal
        then names the configuration instead).
        """
        source = Path(budget_window.record.path)
        old = tmp_path / "old.h5"
        _copy_run(source, old)
        _strip_leftover(old)
        new_frames = RunFile(source).read_frames()
        old_frames = RunFile(old).read_frames()
        assert old_frames.steps.tolist() == new_frames.steps.tolist()
        for mine, theirs in zip(old_frames.budgets, new_frames.budgets, strict=True):
            for a, b in zip(mine, theirs, strict=True):
                assert a.leftover is None
                if b.current is None:
                    continue
                assert a.residual() == pytest.approx(
                    b.residual() + b.leftover, rel=1e-12, abs=1e-9
                )
        assert all(b.leftover is None for b in RunFile(old).read_checkpoint().budgets)
        later = _with_t_end(budget_window.config, 5.0)
        before = old.read_bytes()
        with pytest.raises(ValueError, match="leftover accumulated before the restart"):
            run(later, restart=old, provenance=PROVENANCE)
        assert old.read_bytes() == before
        scenario = _scenario_file(tmp_path, later.merged_mapping())
        argv = [str(scenario), "--restart", str(old), "--quiet"]
        assert run_scenario.main(argv) == run_scenario.EXIT_REFUSED
        assert "leftover" in capsys.readouterr().err
        last = {k: fields[-1] for k, fields in new_frames.fields.items()}
        with pytest.raises(ValueError, match="written before"):
            RunFile(old).write_frame(10_000, 9.0, last, new_frames.budgets[-1])
