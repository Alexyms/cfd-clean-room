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
- scripts/run_scenario.py: its exit codes and summary lines.

The coarse fixtures extend the reference scenario files with the grid
overridden to 80x30 and the error_estimate per-cell tolerance set by
ADR-011 G's formula at the base's t_end, the product row of record on that
grid (hash 6f27f27f, 1021 outer iterations, docs/reports/probe62/hash62.py).
The event times of the transport-only and door fixtures are scaled down so
the suite marches seconds, not minutes; their segments, sources and initial
fields are the reference files'.
"""

import hashlib
import math
import sys
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

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
from src.config import SimConfig
from src.io_manager import RunFile
from src.mesh import SOLID, Mesh
from src.particles import ParticlePhysics
from src.scenarios import Interval, ScenarioManager
from src.solver_transport import TransportSolver
from src.staggered import FaceVelocities
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
    air_change_rate,
    flow_inputs,
    plan_interval,
    run,
    sensor_cells,
    smallest_stable_dt,
)
from tests.test_io_manager import (
    _same_budgets,
    _same_recovery,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
COARSE_GRID = {"nx": 80, "ny": 30}
# ADR-011 G's per-cell bound, 1e-4 rho V_min / t_end, on 80x30 at the base's
# t_end of 60 s, as the product row of record on this grid ran it.
COARSE_MASS_IMBALANCE_TOL = 1.0e-4 * 1.2 * (0.1 * 0.1) / 60.0
COARSE_FACE_HASH = "6f27f27f3e89963a3bee283be46599381cce35d4ae5370db79de71f990dd6e3e"
COARSE_OUTER = 1021
# The open door's state warm-started from the closed one on 80x30, as prompt
# 64 first solved it (results/builder64/door.h5): the warm start's record.
OPEN_FACE_HASH = "17fa707443858ce6e41c3a89547858f4ebad00cdca4fc25d75113eee065ad617"
OPEN_OUTER = 928
# REQ-T05: particle mass conserved to 0.01%. The march closes to about 1e-7
# here, set by the implicit diffusion's diffusion_tol of 1e-8 per step.
REQ_T05 = 1.0e-4
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


def face_hash(u: np.ndarray, v: np.ndarray) -> str:
    """SHA-256 over u's bytes then v's, as the hashes of record are taken."""
    return hashlib.sha256(
        np.ascontiguousarray(u, dtype="<f8").tobytes()
        + np.ascontiguousarray(v, dtype="<f8").tobytes()
    ).hexdigest()


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
        _same_budgets(x, y)
    ra, rb = a.read_sensor_readings(), b.read_sensor_readings()
    assert ra.names == rb.names
    assert ra.steps.tolist() == rb.steps.tolist()
    assert ra.times.tobytes() == rb.times.tobytes()
    assert ra.readings.tobytes() == rb.readings.tobytes()


# ---------------------------------------------------------------------------
# The coarse runs, shared
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def clean_out(tmp_path_factory: pytest.TempPathFactory) -> SimpleNamespace:
    """The coarse clean-out from rest to its hold: the kept base state for the rest."""
    folder = tmp_path_factory.mktemp("clean_out")
    config = SimConfig.from_dict(coarse_raw("scenario_clean_out.yaml"))
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
        """Defects caught: a run key or the transport section counted, so a kept
        state is refused for a run that differs only in them; a solver key the
        solve reads left out, so a state of other settings is reused."""
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
        assert mapping["solver"]["t_end"] == 120.0


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
        _same_recovery(
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
        for existing in (tmp_path / "x.h5", tmp_path / "y.checkpoint.h5"):
            existing.write_bytes(b"")
        for name in ("x.h5", "y.h5"):
            with pytest.raises(ValueError, match="exists; a new run writes new files"):
                run(config, tmp_path / name)
        assert not (tmp_path / "a.h5").exists()

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
        """One state solved from rest, the 80x30 product row's faces bitwise.

        Defects caught: the first state warm-started from something; the run
        file holding a state other than the solve's.
        """
        record = clean_out.record
        assert [f.origin for f in record.flows] == [REST]
        flow = record.flows[0]
        assert (flow.outer, flow.stop_reason) == (
            COARSE_OUTER,
            "error_estimate_and_continuity",
        )
        state = clean_out.run.read_flow_state(flow.fingerprint).state
        assert face_hash(state.u, state.v) == COARSE_FACE_HASH
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
        _same_recovery(clean_out.run.read_recovery(), recovery)

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
        _same_budgets(frames.budgets[-1], record.budgets)
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

        The open state is the warm start of record, faces and outer count.
        Defects caught: the second opening solved again; the open state
        started from rest (its faces and count are the rest solve's, though
        the label says warm); a state warm-started from a state other than
        the one before it.
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
        states = [door.run.read_flow_state(key).state for key in (closed, opened)]
        assert [face_hash(x.u, x.v) for x in states] == [
            COARSE_FACE_HASH,
            OPEN_FACE_HASH,
        ]
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
            _same_budgets(tuple(transport.budget), after.budgets)
            for x, y in zip(before.budgets, after.budgets, strict=True):
                assert x.initial == y.initial
        opened = door.steps[door.record.intervals[0].steps]
        assert all(b.inflow > 0.0 for b in opened.budgets)


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
    """The clean-out with the door open from 1 s to 2 s, to 4 s: a switch and a tracker.

    The hold of one air change is longer than the run, so it is left out.
    """
    raw = coarse_raw("scenario_clean_out.yaml")
    del raw["scenario"]["hold_time"]
    door_raw = coarse_raw("scenario_door_leak.yaml")
    opening = door_raw["scenario"]["events"][0]
    opening["start"], opening["duration"] = 1.0, 1.0
    raw["scenario"]["events"].append(opening)
    raw["solver"]["t_end"] = 4.0
    if cadence is not None:
        raw["solver"]["checkpoint_interval"] = cadence
    return SimConfig.from_dict(raw)


@pytest.mark.integration
class TestRestart:
    def test_after_a_switch_the_restart_is_the_uninterrupted_run(
        self, tmp_path: Path, door: SimpleNamespace
    ) -> None:
        """Killed between two checkpoints after the door opened, restarted, finished.

        The fields, frames, budgets, sensor readings and recovery outputs,
        the local field included, are the uninterrupted run's bitwise.
        Checkpoints every 30 steps, frames every 100: the kill comes after
        frame 100, written after checkpoint 90, which follows the switch.
        Defects caught: the readings held for the next frame not flushed
        before a checkpoint (a gap at the restart); the tracker's state not
        carried; the restart resuming in the first interval's plan; the
        frame and readings written after the checkpoint kept (frame 100 is
        refused on the way back); the budgets not restored.
        """
        config = _door_clean_out(30)
        kept = door.record.path
        whole = run(
            config, tmp_path / "whole.h5", flows_from=kept, provenance=PROVENANCE
        )
        switch = whole.intervals[0].steps
        frame, checkpoint = 100, 90
        assert switch < checkpoint < frame < whole.steps
        cut = tmp_path / "cut.h5"
        with pytest.raises(_KilledError):
            run(
                config,
                cut,
                flows_from=kept,
                on_step=killer(frame + 7),
                provenance=PROVENANCE,
            )
        assert RunFile(cut).read_checkpoint().step == checkpoint
        assert RunFile(cut).read_frames().steps.tolist() == [0, frame]
        again = run(config, restart=cut)
        assert again.restarted_from == checkpoint
        assert [f.origin for f in again.flows] == [RUN_FILE, RUN_FILE]
        assert (again.steps, again.t_final) == (whole.steps, whole.t_final)
        same_frames(RunFile(cut), RunFile(whole.path))
        _same_budgets(again.budgets, whole.budgets)
        _same_recovery(again.recovery, whole.recovery)
        _same_recovery(RunFile(cut).read_recovery(), whole.recovery)

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
        _same_recovery(again.recovery, whole.recovery)

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
        raw = config.merged_mapping()
        raw["solver"]["t_end"] = 4.5
        later = SimConfig.from_dict(raw)
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
        assert face_hash(state.u, state.v) == COARSE_FACE_HASH
        theirs = clean_out.run.read_flow_state(key).state
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
