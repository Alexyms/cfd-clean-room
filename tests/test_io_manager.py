"""The run file and its checkpoint file (src/io_manager.py, ADR-014 D, steps 0 and 3).

What is pinned, with the planted defect each test was shown to catch in its
docstring:

- Each group written and read back is what was written, bitwise: /config,
  a flow state, the transport frames and budgets, the sensor readings, the
  march's checkpoint and the steady checkpoint.
- The steady restart: a solve checkpointed by the periodic writer and killed
  after it resumes from the checkpoint file, in a fresh solver with a larger
  cap, to bitwise what one uninterrupted solve reaches.
- The march restart: a march on a vented room with a recirculating pair,
  killed after its checkpoint, restored from the files in a fresh solver and
  stepped on, ends with the fields, budgets, frames and sensor readings of
  the uninterrupted march, bitwise. The class fields are all the discharge
  needs: it reads its intake from the field the step starts from.
- A restart under a changed key is refused, naming the key; the keys a
  continuation changes (cap, t_end, checkpoint cadence) are not; a resume
  under another prescribed eddy viscosity is refused.
- Crash safety: a failure after the temporary checkpoint file is written and
  before the replace leaves the previous checkpoints and the run file's frames
  readable; a temporary file an earlier crash left is overwritten, not read.
- Prompt 64: the restart check reads the file under the excluded keys in
  force now; a clean-out's recovery outputs round-trip through /recovery/,
  which neither discard touches; the march checkpoint carries named arrays
  (the recovery tracker's state); discard_march drops /transport/ alone.
"""

import re
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import yaml

from src.boundary_concentration import ConcentrationBoundary
from src.boundary_registry import BoundaryRegistry
from src.boundary_staggered import StaggeredBoundary
from src.config import SimConfig
from src.io_manager import (
    BUDGET_TERMS,
    CHECKPOINT,
    RESTART_EXCLUDED_KEYS,
    STEADY_CHECKPOINT,
    RecoveryRecord,
    RunFile,
    config_text,
    restart_hash,
)
from src.mesh import SOLID, Mesh
from src.particles import ParticlePhysics
from src.scenarios import flow_fingerprint
from src.solver_staggered import FlowState, StaggeredSolver, SteadyCheckpoint
from src.solver_transport import DEPOSIT_SURFACES, MassBudget, TransportSolver
from src.staggered import FaceVelocities
from tests.test_warm_start import (
    FIELDS,
    FIXTURES,
    RAW_FIXTURES,
    _channel,
    _same_bits,
    _solved,
    _solver,
)
from tests.vent_cases import DISCHARGE, INTAKE, vent_raw

FINGERPRINT = "0123456789abcdef" * 4
OTHER_FINGERPRINT = "fedcba9876543210" * 4
PROVENANCE = {
    "commit": "c6d2f13",
    "started": "2026-10-17T12:00:00",
    "machine": "test-machine",
    "concurrent": 0,
}


class _KilledError(RuntimeError):
    """The process killed at a chosen point, as a test raises it."""


def _same_checkpoint(a: SteadyCheckpoint, b: SteadyCheckpoint) -> None:
    """Every array, series, count and digest of two checkpoints, bit for bit."""
    for name in FIELDS:
        x, y = getattr(a.state, name), getattr(b.state, name)
        assert (x is None) == (y is None), name
        if x is not None:
            assert x.dtype == y.dtype == np.float64
            assert x.shape == y.shape and x.tobytes() == y.tobytes(), name
    assert (a.state.stop_reason, a.state.outer) == (b.state.stop_reason, b.state.outer)
    for name in ("residual_history", "velocity_steps", "viscosity_steps"):
        assert getattr(a, name) == getattr(b, name), name
    assert (a.rule_version, a.pressure_cap_hits, a.eddy_viscosity_sha256) == (
        b.rule_version,
        b.pressure_cap_hits,
        b.eddy_viscosity_sha256,
    )


def _same_budgets(a: tuple[MassBudget, ...], b: tuple[MassBudget, ...]) -> None:
    """Two sequences of budgets, every term bit for bit, None as None."""
    assert len(a) == len(b)
    for x, y in zip(a, b, strict=True):
        for name in ("initial", "inflow", "outflow", "source", "current"):
            p, q = getattr(x, name), getattr(y, name)
            assert (p is None) == (q is None), name
            if p is not None:
                assert np.float64(p).tobytes() == np.float64(q).tobytes(), name
        assert list(x.deposited) == list(y.deposited) == list(DEPOSIT_SURFACES)
        for surface in DEPOSIT_SURFACES:
            assert (
                np.float64(x.deposited[surface]).tobytes()
                == np.float64(y.deposited[surface]).tobytes()
            ), surface


def _budget(seed: int, *, fresh: bool = False) -> MassBudget:
    """A budget with a distinct value in every term; FRESH has no field yet."""
    rng = np.random.default_rng(seed)
    values = rng.random(9)
    return MassBudget(
        initial=None if fresh else float(values[0]),
        inflow=float(values[1]),
        outflow=float(values[2]),
        source=float(values[3]),
        deposited={
            s: float(v) for s, v in zip(DEPOSIT_SURFACES, values[4:8], strict=True)
        },
        current=None if fresh else float(values[8]),
    )


def _fields(seed: int, classes: int = 2) -> dict[int, np.ndarray]:
    """Class fields of a 3 by 4 grid that differ in every cell."""
    rng = np.random.default_rng(seed)
    return {k: rng.random((3, 4)) for k in range(classes)}


def _steady(
    turbulent: bool, reason: str | None = "max_simple_iter"
) -> SteadyCheckpoint:
    """A checkpoint of a 4 by 3 grid with every series and count set."""
    rng = np.random.default_rng(60)
    arrays = {
        "u": rng.normal(size=(3, 5)),
        "v": rng.normal(size=(4, 4)),
        "p": rng.normal(size=(3, 4)),
    }
    if turbulent:
        arrays |= {name: rng.random((3, 4)) for name in ("k", "eps", "nu_t")}
    state = FlowState(**arrays, stop_reason=reason, outer=2)
    window = (0.1, 1e-300) if turbulent else ()
    return SteadyCheckpoint(
        state,
        (0.5, 0.25),
        (3e-7, 2e-7),
        window,
        5 if turbulent else None,
        1,
        None if turbulent else "0123456789abcdef" * 4,
    )


# ---------------------------------------------------------------------------
# Round trips
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestConfig:
    def test_what_is_written_is_what_is_read(self, tmp_path: Path) -> None:
        """The mapping, its two hashes and the provenance come back as written.

        Defects caught: the text written with sorted keys (a boundary
        segment's place in the order lost); the restart hash taken over the
        whole mapping; a provenance field dropped.
        """
        raw = _room_raw()
        run = RunFile(tmp_path / "run.h5")
        run.write_config(raw, **PROVENANCE)
        stored = run.read_config()
        assert stored.mapping == raw
        assert list(stored.mapping["boundaries"]) == list(raw["boundaries"])
        assert stored.restart_sha256 == restart_hash(raw)
        whole = {**raw, "solver": dict(raw["solver"])}
        whole["solver"]["max_simple_iter"] += 1
        assert restart_hash(whole) == restart_hash(raw)
        assert config_text(whole) != config_text(raw)
        assert (stored.commit, stored.started, stored.machine, stored.concurrent) == (
            PROVENANCE["commit"],
            PROVENANCE["started"],
            PROVENANCE["machine"],
            PROVENANCE["concurrent"],
        )
        assert SimConfig.from_dict(stored.mapping).nx == raw["domain"]["nx"]

    def test_a_second_configuration_is_refused(self, tmp_path: Path) -> None:
        """Defect caught: a restart overwriting the configuration it is checked against."""
        run = RunFile(tmp_path / "run.h5")
        run.write_config(_room_raw(), **PROVENANCE)
        with pytest.raises(ValueError, match="holds a configuration already"):
            run.write_config(_room_raw(), **PROVENANCE)

    @pytest.mark.parametrize(
        ("change", "error"),
        [
            ({"concurrent": True}, TypeError),
            ({"concurrent": 1.0}, TypeError),
            ({"concurrent": -1}, ValueError),
            ({"commit": None}, TypeError),
        ],
    )
    def test_bad_provenance_is_refused(
        self, tmp_path: Path, change: dict, error: type
    ) -> None:
        """Defect caught: a bool process count stored as 1, a None commit as text."""
        with pytest.raises(error):
            RunFile(tmp_path / "run.h5").write_config(
                _room_raw(), **(PROVENANCE | change)
            )


@pytest.mark.unit
class TestFlowState:
    @pytest.mark.parametrize("turbulent", [False, True])
    def test_what_is_written_is_what_is_read(
        self, tmp_path: Path, turbulent: bool
    ) -> None:
        """Faces, pressure, turbulence, the stop and the timings, bitwise.

        Defects caught: a field written as float32; k, eps or nu_t dropped; the
        stage seconds or rule readings not written.
        """
        state = _steady(turbulent).state
        run = RunFile(tmp_path / "run.h5")
        stages = {"momentum": 1.5, "pressure": 2.25, "correct": 0.125}
        readings = {"estimate": 3.0e-7, "imbalance": float("inf")}
        run.write_flow_state(
            FINGERPRINT,
            state,
            wall_seconds=4.0,
            stage_seconds=stages,
            rule_readings=readings,
        )
        record = run.read_flow_state(FINGERPRINT)
        for name in FIELDS:
            x, y = getattr(record.state, name), getattr(state, name)
            assert (x is None) == (y is None), name
            if x is not None:
                assert x.tobytes() == y.tobytes(), name
        assert (record.state.stop_reason, record.state.outer) == (
            state.stop_reason,
            state.outer,
        )
        assert record.wall_seconds == 4.0
        assert record.stage_seconds == stages and record.rule_readings == readings
        assert run.flow_fingerprints() == (FINGERPRINT,)

    def test_a_state_written_twice_or_under_a_bad_key_is_refused(
        self, tmp_path: Path
    ) -> None:
        """Defects caught: a flow state overwritten; a key that is not a SHA-256."""
        run = RunFile(tmp_path / "run.h5")
        state = _steady(False).state
        keys = {"wall_seconds": 1.0, "stage_seconds": {}, "rule_readings": {}}
        run.write_flow_state(FINGERPRINT, state, **keys)
        with pytest.raises(ValueError, match="holds flow state"):
            run.write_flow_state(FINGERPRINT, state, **keys)
        with pytest.raises(ValueError, match="64 lowercase hexadecimal digits"):
            run.write_flow_state("a/b", state, **keys)
        with pytest.raises(TypeError, match="wall_seconds must be a real number"):
            run.write_flow_state(
                OTHER_FINGERPRINT, state, **(keys | {"wall_seconds": True})
            )
        assert RunFile(tmp_path / "none.h5").flow_fingerprints() == ()


@pytest.mark.unit
class TestFrames:
    def _write(self, run: RunFile) -> list[tuple]:
        frames = [
            (0, 0.0, _fields(1), [_budget(1, fresh=True), _budget(2, fresh=True)]),
            (5, 0.125, _fields(2), [_budget(3), _budget(4)]),
            (10, 0.25, _fields(3), [_budget(5), _budget(6)]),
        ]
        for frame in frames:
            run.write_frame(*frame)
        return frames

    def test_what_is_written_is_what_is_read(self, tmp_path: Path) -> None:
        """Steps, times, every class's frames and every budget, bitwise.

        The first frame's budgets have no field yet (initial and current
        None). Defects caught: None read back as NaN or zero; the budgets of
        two classes swapped; a frame appended over the last one.
        """
        run = RunFile(tmp_path / "run.h5")
        frames = self._write(run)
        read = run.read_frames()
        assert read.steps.dtype == np.int64 and read.steps.tolist() == [0, 5, 10]
        assert read.times.tobytes() == np.array([0.0, 0.125, 0.25]).tobytes()
        assert sorted(read.fields) == [0, 1]
        for k in (0, 1):
            stacked = np.stack([frame[2][k] for frame in frames])
            assert read.fields[k].tobytes() == stacked.tobytes()
        for got, frame in zip(read.budgets, frames, strict=True):
            _same_budgets(got, tuple(frame[3]))
        with h5py.File(run.path, "r") as handle:
            assert list(handle["transport/budgets"].attrs["terms"]) == list(
                BUDGET_TERMS
            )

    def test_a_frame_out_of_order_or_of_other_classes_is_refused(
        self, tmp_path: Path
    ) -> None:
        """A frame out of order, of other classes or shapes, or a bad step or time.

        Defects caught: a restart's frames appended after the crash's; a class
        lost; a frame of another grid appended; a bool time stored as 1.0.
        """
        run = RunFile(tmp_path / "run.h5")
        self._write(run)
        with pytest.raises(ValueError, match="not above the last frame's"):
            run.write_frame(10, 0.25, _fields(4), [_budget(1), _budget(2)])
        with pytest.raises(ValueError, match="each class"):
            run.write_frame(15, 0.375, _fields(4), [_budget(1)])
        with pytest.raises(ValueError, match="the file's"):
            run.write_frame(15, 0.375, _fields(4, 3), [_budget(k) for k in range(3)])
        with pytest.raises(TypeError, match="step must be an int"):
            run.write_frame(True, 0.375, _fields(4), [_budget(1), _budget(2)])
        with pytest.raises(TypeError, match="t must be a real number"):
            run.write_frame(15, True, _fields(4), [_budget(1), _budget(2)])
        wider = {k: np.zeros((3, 5)) for k in (0, 1)}
        with pytest.raises(ValueError, match=r"has shape \(3, 5\) and the file's"):
            run.write_frame(15, 0.375, wider, [_budget(1), _budget(2)])
        flat = {k: np.zeros(12) for k in (0, 1)}
        with pytest.raises(ValueError, match="must be 2-D"):
            run.write_frame(15, 0.375, flat, [_budget(1), _budget(2)])
        assert run.read_frames().steps.tolist() == [0, 5, 10]

    def test_sensor_readings_are_appended_and_read_back(self, tmp_path: Path) -> None:
        """Two batches come back as one series, bitwise.

        Defects caught: a batch written over the last; readings appended
        under other sensor names.
        """
        run = RunFile(tmp_path / "run.h5")
        rng = np.random.default_rng(63)
        first, second = rng.random((3, 2, 2)), rng.random((2, 2, 2))
        names = ["centre", "floor"]
        run.write_sensor_readings(np.arange(1, 4), np.arange(1, 4) * 0.1, first, names)
        run.write_sensor_readings(np.arange(4, 6), np.arange(4, 6) * 0.1, second, names)
        read = run.read_sensor_readings()
        assert read.names == ("centre", "floor")
        assert read.steps.tolist() == [1, 2, 3, 4, 5]
        assert read.readings.tobytes() == np.concatenate([first, second]).tobytes()
        with pytest.raises(ValueError, match="sensors are"):
            run.write_sensor_readings(
                np.array([6]), np.array([0.6]), second[:1], ["a", "b"]
            )
        with pytest.raises(ValueError, match="not above the last reading's"):
            run.write_sensor_readings(np.array([5]), np.array([0.5]), second[:1], names)

    def test_discard_after_keeps_what_the_checkpoint_covers(
        self, tmp_path: Path
    ) -> None:
        """Frames and readings after the step go; those at it stay; appending resumes.

        Defect caught: the frames a crash left after the checkpoint kept, so the
        restart's first frame is refused or written twice.
        """
        run = RunFile(tmp_path / "run.h5")
        frames = self._write(run)
        run.write_sensor_readings(
            np.arange(1, 11), np.arange(1, 11) * 0.025, np.ones((10, 2, 1)), ["centre"]
        )
        run.discard_after(5)
        read = run.read_frames()
        assert read.steps.tolist() == [0, 5]
        assert read.fields[1].shape == (2, 3, 4)
        assert len(read.budgets) == 2
        assert run.read_sensor_readings().steps.tolist() == [1, 2, 3, 4, 5]
        run.write_frame(*frames[2])
        assert run.read_frames().steps.tolist() == [0, 5, 10]


@pytest.mark.unit
class TestCheckpoints:
    def test_the_march_checkpoint_is_what_was_written(self, tmp_path: Path) -> None:
        """Step, time, fields, budgets and fingerprint, bitwise, in the checkpoint file.

        Defects caught: the time stored as float32 or rebuilt from the step;
        a class field or budget dropped; the checkpoint written into the run
        file, which a crash while it is open can make unreadable.
        """
        run = RunFile(tmp_path / "run.h5")
        fields, budgets = _fields(7), [_budget(7), _budget(8)]
        t = 0.1 + 0.2
        run.checkpoint(12, t, fields, budgets, FINGERPRINT)
        read = run.read_checkpoint()
        assert (read.step, read.fingerprint) == (12, FINGERPRINT)
        assert np.float64(read.t).tobytes() == np.float64(t).tobytes()
        assert sorted(read.fields) == [0, 1]
        for k in (0, 1):
            assert read.fields[k].tobytes() == fields[k].tobytes()
        _same_budgets(read.budgets, tuple(budgets))
        assert run.checkpoint_path == tmp_path / "run.checkpoint.h5"
        assert not run.path.exists()

    @pytest.mark.parametrize("turbulent", [False, True])
    @pytest.mark.parametrize("reason", ["max_simple_iter", None])
    def test_the_steady_checkpoint_is_what_was_written(
        self, tmp_path: Path, turbulent: bool, reason: str | None
    ) -> None:
        """Defects caught: a None stop or rule version read back as a string or 0;
        the eddy viscosity digest or the fingerprint not written."""
        checkpoint = _steady(turbulent, reason)
        run = RunFile(tmp_path / "run.h5")
        run.write_steady_checkpoint(checkpoint, FINGERPRINT)
        fingerprint, read = run.read_steady_checkpoint()
        assert fingerprint == FINGERPRINT
        _same_checkpoint(read, checkpoint)

    def test_the_two_checkpoints_keep_each_other(self, tmp_path: Path) -> None:
        """Each rewrite replaces its own group and copies the other across.

        The coupled steady checkpoint is replaced by the laminar one, so k,
        eps and nu_t must go as well as come. Defects caught: the other
        group not copied into the new file; the old group copied and
        written over, merging two checkpoints.
        """
        run = RunFile(tmp_path / "run.h5")
        run.write_steady_checkpoint(_steady(True), FINGERPRINT)
        run.checkpoint(3, 0.5, _fields(1), [_budget(1), _budget(2)], FINGERPRINT)
        run.write_steady_checkpoint(_steady(False), OTHER_FINGERPRINT)
        run.checkpoint(6, 1.0, _fields(2), [_budget(3), _budget(4)], OTHER_FINGERPRINT)
        fingerprint, steady = run.read_steady_checkpoint()
        assert fingerprint == OTHER_FINGERPRINT
        _same_checkpoint(steady, _steady(False))
        assert run.read_checkpoint().step == 6
        with h5py.File(run.checkpoint_path, "r") as handle:
            assert sorted(handle) == [CHECKPOINT, STEADY_CHECKPOINT]
            assert "k" not in handle[STEADY_CHECKPOINT]

    @pytest.mark.parametrize(
        "dropped", [("k",), ("eps",), ("nu_t",), ("k", "eps"), ("eps", "nu_t")]
    )
    def test_a_group_with_some_of_k_eps_nu_t_is_refused(
        self, tmp_path: Path, dropped: tuple[str, ...]
    ) -> None:
        """A coupled group missing datasets is refused, not read back as laminar.

        Defect caught: the read keyed on all three being present, so a damaged
        coupled checkpoint came back as a laminar one without an error.
        """
        run = RunFile(tmp_path / "run.h5")
        run.write_steady_checkpoint(_steady(True), FINGERPRINT)
        with h5py.File(run.checkpoint_path, "a") as handle:
            for name in dropped:
                del handle[STEADY_CHECKPOINT][name]
        with pytest.raises(ValueError, match="given together or not at all"):
            run.read_steady_checkpoint()

    def test_no_checkpoint_is_refused_and_bad_arguments_too(
        self, tmp_path: Path
    ) -> None:
        """Defects caught: an empty checkpoint returned when none was written; a
        FlowState written as a checkpoint; a march checkpoint without its flow
        key, with a bool or None time, a flat field or a budget that is not one
        (each refused before the checkpoint file is touched)."""
        run = RunFile(tmp_path / "run.h5")
        with pytest.raises(KeyError, match="no checkpoint was written"):
            run.read_checkpoint()
        run.write_steady_checkpoint(_steady(False), FINGERPRINT)
        with pytest.raises(KeyError, match="holds no /checkpoint/ group"):
            run.read_checkpoint()
        with pytest.raises(TypeError, match="checkpoint must be a SteadyCheckpoint"):
            run.write_steady_checkpoint(_steady(False).state, FINGERPRINT)  # type: ignore[arg-type]
        budgets = [_budget(1), _budget(2)]
        with pytest.raises(ValueError, match="64 lowercase hexadecimal digits"):
            run.checkpoint(1, 0.1, _fields(1), budgets, "")
        with pytest.raises(TypeError, match="t must be a real number"):
            run.checkpoint(1, None, _fields(1), budgets, FINGERPRINT)
        with pytest.raises(ValueError, match="must be 2-D"):
            run.checkpoint(
                1, 0.1, {k: np.zeros(12) for k in (0, 1)}, budgets, FINGERPRINT
            )
        with pytest.raises(TypeError, match="must be a MassBudget"):
            run.checkpoint(1, 0.1, _fields(1), [budgets[0], {}], FINGERPRINT)
        with pytest.raises(KeyError, match="holds no /checkpoint/ group"):
            run.read_checkpoint()


# ---------------------------------------------------------------------------
# The steady restart
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def uninterrupted() -> dict:
    """Each warm-start fixture solved from rest to its rule's stop."""
    return {name: _solved(make()) for name, make in FIXTURES.items()}


@pytest.mark.integration
class TestSteadyRestart:
    @pytest.mark.parametrize("name", list(FIXTURES))
    def test_a_killed_solve_resumes_from_the_file_to_the_uninterrupted_solve(
        self, name: str, tmp_path: Path, uninterrupted: dict
    ) -> None:
        """Checkpointed every 50 outer, killed 40 later, resumed under a larger cap.

        The kill falls 10 outer iterations before the last multiple of 50
        below the stop, so the checkpoint read back is the one before it,
        within the rule's window of the stop (where a window not restored
        cannot fill again in time). The restart's configuration differs from
        the killed run's in its cap only, and is accepted. Defects caught: a
        field written as float32; the rule's window or the residual history
        not written; the checkpoint not rewritten periodically (the file
        holds the first one); the cap in the restart hash.
        """
        make_raw = RAW_FIXTURES[name]
        whole = uninterrupted[name]
        stop = len(whole.residual_history)
        kill = (stop // 50) * 50 - 10
        killed_raw = make_raw(cap=kill + 5, interval=50)
        config = SimConfig.from_dict(killed_raw)
        key = flow_fingerprint(config)
        run = RunFile(tmp_path / "run.h5")
        run.write_config(killed_raw, **PROVENANCE)

        def killed_at(state: object) -> None:
            if state.iteration + 1 == kill:
                raise _KilledError

        with pytest.raises(_KilledError):
            _solver(config).solve_steady(
                killed_at,
                on_checkpoint=partial(run.write_steady_checkpoint, fingerprint=key),
            )

        restart = RunFile(tmp_path / "run.h5")
        resumed_raw = make_raw(interval=50)
        restart.check_restart(resumed_raw)
        fingerprint, checkpoint = restart.read_steady_checkpoint()
        assert fingerprint == key
        assert checkpoint.state.outer == (kill // 50) * 50
        resumed = _solved(SimConfig.from_dict(resumed_raw), resume=checkpoint)
        _same_bits(resumed, whole)
        assert resumed.converged

    def test_a_resume_under_another_eddy_viscosity_is_refused(
        self, tmp_path: Path
    ) -> None:
        """The digest survives the file, and a resume under another field is refused.

        Defects caught: the digest not written (read back as None, so a
        resume under no field runs); the solver's comparison removed.
        """
        mesh = Mesh(_channel())
        field = np.full(mesh.cell_type.shape, 2.0e-5)
        run = RunFile(tmp_path / "run.h5")
        checkpoint = _solved(_channel(cap=20), eddy_viscosity=field).steady_checkpoint
        run.write_steady_checkpoint(checkpoint, FINGERPRINT)
        _, read = run.read_steady_checkpoint()
        assert read.eddy_viscosity_sha256 == checkpoint.eddy_viscosity_sha256
        for given in (1.5 * field, None):
            solver = _solver(_channel(cap=40))
            with pytest.raises(ValueError, match="under the eddy_viscosity it ran"):
                solver.solve_steady(eddy_viscosity=given, resume=read)
            assert solver.residual_history == []
        resumed = _solved(_channel(cap=40), eddy_viscosity=field, resume=read)
        _same_bits(resumed, _solved(_channel(cap=40), eddy_viscosity=field))


# ---------------------------------------------------------------------------
# The march restart
# ---------------------------------------------------------------------------

STEPS = 50
KILL = 27
VENTS = {"discharge": {**DISCHARGE, "recirculate_from": "intake"}, "intake": INTAKE}


def _room_raw() -> dict:
    """tests/vent_cases.py's room with a recirculating pair, two classes, two sensors.

    Air 55 times as viscous under error_estimate, as the vent transport
    tests solve it (tests/test_solver_transport.py); frames every 5 steps,
    checkpoints every 10.
    """
    raw = vent_raw(VENTS)
    raw["fluid"]["viscosity"] = 1.0e-3
    raw["particles"]["sizes"] = [1.0e-6, 5.0e-6]
    raw["particles"]["hepa_reference"] = {
        "diameters": [1.0e-6, 5.0e-6],
        "efficiencies": [0.9999, 0.99999],
    }
    raw["sensors"].append({"name": "floor", "x": 0.3, "y": 0.1})
    raw["solver"].update(
        stopping_rule="error_estimate",
        max_simple_iter=3000,
        iteration_error_tol=1.0e-6,
        mass_imbalance_tol=1.0e-10,
        output_interval=5,
        checkpoint_interval=10,
    )
    return raw


def _transport(config: SimConfig, mesh: Mesh) -> TransportSolver:
    physics = ParticlePhysics(config)
    boundary = ConcentrationBoundary(mesh, config, physics, BoundaryRegistry(config))
    return TransportSolver(mesh, config, physics, boundary)


@pytest.fixture(scope="module")
def room() -> SimpleNamespace:
    """The room's flow solved, its step, sources, sensors and initial fields."""
    raw = _room_raw()
    config = SimConfig.from_dict(raw)
    mesh = Mesh(config)
    solver = StaggeredSolver(mesh, config, StaggeredBoundary(mesh, config))
    solver.solve_steady()
    assert solver.converged
    state = solver.flow_state
    faces = FaceVelocities.copy_of(np.array(state.u), np.array(state.v))
    classes = range(len(config.particle_sizes))
    probe = _transport(config, mesh)
    dt = 0.9 * min(probe.stable_dt(faces, k) for k in classes)
    live = mesh.cell_type != SOLID
    x, y = np.meshgrid(mesh.xc, mesh.yc)
    sources = {}
    for k in classes:
        rate = np.zeros(mesh.cell_type.shape)
        rate[5:7, 2:4] = 2.0e3 / (k + 1)
        assert live[5:7, 2:4].all()
        sources[k] = rate
    return SimpleNamespace(
        raw=raw,
        config=config,
        mesh=mesh,
        state=state,
        stage_seconds=dict(solver.stage_seconds),
        fingerprint=flow_fingerprint(config),
        dt=dt,
        sources=sources,
        initial={
            k: np.where(live, 100.0 + 40.0 * x + (20.0 + k) * y, 0.0) for k in classes
        },
        sensors=[
            (
                int(np.argmin(np.abs(mesh.yc - s.y))),
                int(np.argmin(np.abs(mesh.xc - s.x))),
            )
            for s in config.sensors
        ],
        names=[s.name for s in config.sensors],
    )


def _start(
    run: RunFile, room: SimpleNamespace
) -> tuple[TransportSolver, FaceVelocities]:
    """A run's first writes: the configuration, the flow state and frame 0."""
    run.write_config(room.raw, **PROVENANCE)
    run.write_flow_state(
        room.fingerprint,
        room.state,
        wall_seconds=1.0,
        stage_seconds=room.stage_seconds,
        rule_readings={},
    )
    solver = _transport(room.config, room.mesh)
    run.write_frame(0, 0.0, room.initial, solver.budget)
    faces = FaceVelocities.copy_of(np.array(room.state.u), np.array(room.state.v))
    return solver, faces


def _march(
    run: RunFile,
    room: SimpleNamespace,
    solver: TransportSolver,
    faces: FaceVelocities,
    fields: dict[int, np.ndarray],
    step: int,
    t: float,
    *,
    kill: int | None = None,
) -> tuple[dict[int, np.ndarray], float]:
    """Step every class to STEPS as step 4's loop will: frames, readings, checkpoints.

    The time is accumulated, so a restart that rebuilt it from the step
    would not reproduce it. Raises _KilledError after step KILL's writes.
    """
    config = room.config
    pending: list[tuple[int, float, list[list[float]]]] = []
    while step < STEPS:
        fields = {
            k: solver.solve_timestep(c, faces, k, room.dt, sources=room.sources[k])
            for k, c in fields.items()
        }
        step += 1
        t += room.dt
        pending.append(
            (
                step,
                t,
                [[float(c[j, i]) for j, i in room.sensors] for c in fields.values()],
            )
        )
        if step % config.output_interval == 0:
            run.write_frame(step, t, fields, solver.budget)
            run.write_sensor_readings(
                np.array([p[0] for p in pending]),
                np.array([p[1] for p in pending]),
                np.array([p[2] for p in pending]),
                room.names,
            )
            pending = []
        if step % config.checkpoint_interval == 0:
            run.checkpoint(step, t, fields, solver.budget, room.fingerprint)
        if step == kill:
            raise _KilledError
    return fields, t


@pytest.mark.integration
class TestMarchRestart:
    def test_a_killed_march_restored_from_the_files_is_the_uninterrupted_march(
        self, tmp_path: Path, room: SimpleNamespace
    ) -> None:
        """Killed at step 27, restored from step 20's checkpoint, stepped on to 50.

        The restart reads its flow state from the run file and its fields,
        budgets, step and time from the checkpoint file, in a fresh
        transport solver, discards the frame and readings written after step
        20, and steps on. The discharge recirculates the intake's air (the
        supply is clean, so every particle that enters comes back through
        it), and nothing about it is carried but the fields. Defects caught:
        the budgets not restored; the time rebuilt from the step; a class
        field written as float32; the frames after the checkpoint kept.
        """
        whole = RunFile(tmp_path / "whole.h5")
        solver, faces = _start(whole, room)
        fields, t = _march(whole, room, solver, faces, dict(room.initial), 0, 0.0)
        assert all(b.inflow > 0.0 for b in solver.budget)

        killed = RunFile(tmp_path / "run.h5")
        k_solver, k_faces = _start(killed, room)
        with pytest.raises(_KilledError):
            _march(
                killed, room, k_solver, k_faces, dict(room.initial), 0, 0.0, kill=KILL
            )
        assert killed.read_frames().steps[-1] == 25

        restart = RunFile(tmp_path / "run.h5")
        later = {**room.raw, "solver": {**room.raw["solver"], "t_end": 2.0}}
        restart.check_restart(later)
        checkpoint = restart.read_checkpoint()
        assert checkpoint.step == 20
        record = restart.read_flow_state(checkpoint.fingerprint)
        r_faces = FaceVelocities.copy_of(
            np.array(record.state.u), np.array(record.state.v)
        )
        r_solver = _transport(room.config, room.mesh)
        r_solver.restore_budgets(checkpoint.budgets)
        restart.discard_after(checkpoint.step)
        r_fields, r_t = _march(
            restart,
            room,
            r_solver,
            r_faces,
            checkpoint.fields,
            checkpoint.step,
            checkpoint.t,
        )

        assert np.float64(r_t).tobytes() == np.float64(t).tobytes()
        for k in fields:
            assert r_fields[k].tobytes() == fields[k].tobytes(), k
        _same_budgets(tuple(r_solver.budget), tuple(solver.budget))
        a, b = restart.read_frames(), whole.read_frames()
        assert a.steps.tolist() == b.steps.tolist() == list(range(0, STEPS + 1, 5))
        assert a.times.tobytes() == b.times.tobytes()
        for k in b.fields:
            assert a.fields[k].tobytes() == b.fields[k].tobytes(), k
        for x, y in zip(a.budgets, b.budgets, strict=True):
            _same_budgets(x, y)
        ra, rb = restart.read_sensor_readings(), whole.read_sensor_readings()
        assert ra.steps.tolist() == rb.steps.tolist() == list(range(1, STEPS + 1))
        assert ra.readings.tobytes() == rb.readings.tobytes()
        assert ra.times.tobytes() == rb.times.tobytes()


# ---------------------------------------------------------------------------
# Restart refusals
# ---------------------------------------------------------------------------


def _changed(raw: dict, path: tuple, value: object) -> dict:
    """A deep copy of RAW with the key at PATH set to VALUE (added if absent)."""
    out = yaml_copy(raw)
    holder = out
    for key in path[:-1]:
        holder = holder[key]
    holder[path[-1]] = value
    return out


def yaml_copy(raw: dict) -> dict:
    """A deep copy through the YAML text, as the run file stores the mapping."""
    return yaml.safe_load(config_text(raw))


@pytest.mark.unit
class TestRestartRefusals:
    @pytest.mark.parametrize(
        ("path", "value", "named"),
        [
            (
                ("boundaries", "hepa_supply", "velocity"),
                0.25,
                "boundaries.hepa_supply.velocity",
            ),
            (("domain", "nx"), 32, "domain.nx"),
            (("transport", "cfl_number"), 0.2, "transport.cfl_number"),
            (("solver", "momentum_sweeps"), 2, "solver.momentum_sweeps"),
            (("obstacles", 0, "x_end"), 1.375, "obstacles[0].x_end"),
        ],
    )
    def test_a_changed_key_is_refused_naming_it(
        self, tmp_path: Path, path: tuple, value: object, named: str
    ) -> None:
        """A boundary, a grid and a transport key changed, a solver key added, a list item.

        Defects caught: the restart check removed; the check reading only
        the excluded keys' sections; the message naming a key that did not
        change.
        """
        run = RunFile(tmp_path / "run.h5")
        raw = _room_raw()
        run.write_config(raw, **PROVENANCE)
        with pytest.raises(
            ValueError, match=re.escape(f"restart refused: {named} differs")
        ):
            run.check_restart(_changed(raw, path, value))

    def test_a_removed_key_is_refused_naming_it(self, tmp_path: Path) -> None:
        """Defect caught: a key present in one mapping only compared as equal."""
        run = RunFile(tmp_path / "run.h5")
        raw = _room_raw()
        run.write_config(raw, **PROVENANCE)
        edited = yaml_copy(raw)
        del edited["boundaries"]["hepa_supply"]["velocity"]
        with pytest.raises(ValueError, match=r"hepa_supply\.velocity differs"):
            run.check_restart(edited)

    @pytest.mark.parametrize(
        ("key", "value"),
        [("max_simple_iter", 9000), ("t_end", 2.0), ("checkpoint_interval", 25)],
    )
    def test_the_keys_a_continuation_changes_are_accepted(
        self, tmp_path: Path, key: str, value: object
    ) -> None:
        """The cap, t_end and the cadence may change; the cadence may also go.

        Defect caught: the excluded set emptied, so the continuation every
        capped solve and capped march needs is refused.
        """
        assert ("solver", key) in RESTART_EXCLUDED_KEYS
        run = RunFile(tmp_path / "run.h5")
        raw = _room_raw()
        run.write_config(raw, **PROVENANCE)
        run.check_restart(raw)
        run.check_restart(_changed(raw, ("solver", key), value))
        if key == "checkpoint_interval":
            gone = yaml_copy(raw)
            del gone["solver"][key]
            run.check_restart(gone)

    def test_a_file_without_a_configuration_is_refused(self, tmp_path: Path) -> None:
        """Defect caught: a restart from a file that records no configuration accepted."""
        run = RunFile(tmp_path / "run.h5")
        run.write_frame(0, 0.0, _fields(1), [_budget(1), _budget(2)])
        with pytest.raises(KeyError, match="holds no /config"):
            run.check_restart(_room_raw())

    def test_the_file_is_read_under_the_excluded_keys_in_force_now(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A key added to the set widens an existing file's restart; one removed narrows it.

        Review 63 S2, prompt 64's decision 3. Defect caught: the stored
        restart_sha256 compared, which holds the set of the day the file
        was written, so the widened restart is refused with a message that
        names no key.
        """
        run = RunFile(tmp_path / "run.h5")
        raw = _room_raw()
        run.write_config(raw, **PROVENANCE)
        later = _changed(raw, ("solver", "output_interval"), 7)
        with pytest.raises(ValueError, match=r"solver\.output_interval differs"):
            run.check_restart(later)
        widened = (*RESTART_EXCLUDED_KEYS, ("solver", "output_interval"))
        monkeypatch.setattr("src.io_manager.RESTART_EXCLUDED_KEYS", widened)
        run.check_restart(later)
        narrowed = tuple(k for k in RESTART_EXCLUDED_KEYS if k[1] != "t_end")
        monkeypatch.setattr("src.io_manager.RESTART_EXCLUDED_KEYS", narrowed)
        with pytest.raises(ValueError, match=r"restart refused: solver\.t_end differs"):
            run.check_restart(_changed(raw, ("solver", "t_end"), 2.0))

    def test_a_mapping_that_is_not_one_is_refused(self, tmp_path: Path) -> None:
        """Review 63 S4. Defect caught: a list hashed and refused as a differing key."""
        run = RunFile(tmp_path / "run.h5")
        run.write_config(_room_raw(), **PROVENANCE)
        with pytest.raises(TypeError, match="mapping must be a mapping, got list"):
            run.check_restart([])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# The recovery outputs and the march's carried arrays (prompt 64)
# ---------------------------------------------------------------------------


def _recovery(**changed: object) -> RecoveryRecord:
    """A record of 2 classes, 3 sensors and a 3 by 4 grid, every value distinct."""
    rng = np.random.default_rng(64)
    local = rng.random((2, 3, 4))
    local[0, 0, 0] = np.nan
    local[1, 2, 3] = np.inf
    values = {
        "sensor_names": ("a", "b", "c"),
        "thresholds": np.array([1.0e5, 29.3]),
        "initial": np.array([1.0e7, 2930.0]),
        "hold_time": 5.562,
        "held": True,
        "t_final": 31.25,
        "air_change_rate": 0.18,
        "well_mixed_time": 25.6,
        "room_recovery_time": 25.6875,
        "recovery_time": np.array([20.5, np.nan]),
        "defined": np.array([True, False]),
        "sensor_time": rng.random((2, 3)),
        "mean_time": np.array([27.0, np.nan]),
        "ratio": np.array([27.0 / 25.6, np.nan]),
        "local_recovery": local,
        "local_recovery_max": np.array([0.9, np.inf]),
    }
    values.update(changed)
    return RecoveryRecord(**values)


def _same_recovery(a: RecoveryRecord, b: RecoveryRecord) -> None:
    """Every field of two records, arrays bit for bit."""
    assert a.sensor_names == b.sensor_names
    assert (a.hold_time, a.held) == (b.hold_time, b.held)
    for name in (
        "t_final",
        "air_change_rate",
        "well_mixed_time",
        "room_recovery_time",
    ):
        assert (
            np.float64(getattr(a, name)).tobytes()
            == np.float64(getattr(b, name)).tobytes()
        ), name
    for name in (
        "thresholds",
        "initial",
        "recovery_time",
        "sensor_time",
        "mean_time",
        "ratio",
        "local_recovery",
        "local_recovery_max",
    ):
        x, y = getattr(a, name), getattr(b, name)
        assert x.dtype == y.dtype == np.float64, name
        assert x.shape == y.shape and x.tobytes() == y.tobytes(), name
    assert a.defined.dtype == b.defined.dtype == np.bool_
    assert a.defined.tolist() == b.defined.tolist()


@pytest.mark.unit
class TestRecovery:
    @pytest.mark.parametrize("hold", [5.562, None])
    def test_what_is_written_is_what_is_read(
        self, tmp_path: Path, hold: float | None
    ) -> None:
        """NaN, inf and a None hold survive the round trip, bitwise.

        Defects caught: the hold stored as NaN or 0 when None; defined read
        back as int8; an array stored as float32.
        """
        run = RunFile(tmp_path / "run.h5")
        record = _recovery(hold_time=hold, held=hold is not None)
        run.write_recovery(record)
        _same_recovery(run.read_recovery(), record)

    def test_a_second_write_replaces_the_first(self, tmp_path: Path) -> None:
        """A restart that runs on to a later end writes its own outputs.

        Defect caught: the group kept and the second write refused, or merged.
        """
        run = RunFile(tmp_path / "run.h5")
        run.write_recovery(_recovery())
        later = _recovery(hold_time=None, held=False, t_final=40.0)
        run.write_recovery(later)
        _same_recovery(run.read_recovery(), later)

    def test_discards_leave_it_alone(self, tmp_path: Path) -> None:
        """discard_after and discard_march touch /transport/ only (review 63 S3).

        Defects caught: discard_march deleting the configuration, the flow
        states or the recovery outputs; discard_after resizing them.
        """
        run = RunFile(tmp_path / "run.h5")
        run.write_config(_room_raw(), **PROVENANCE)
        run.write_flow_state(
            FINGERPRINT,
            _steady(False).state,
            wall_seconds=1.0,
            stage_seconds={},
            rule_readings={},
        )
        for step in (0, 5, 10):
            run.write_frame(step, step * 0.1, _fields(step), [_budget(1), _budget(2)])
        run.write_recovery(_recovery())
        run.discard_after(5)
        _same_recovery(run.read_recovery(), _recovery())
        run.discard_march()
        with pytest.raises(KeyError, match="holds no /transport/ frames"):
            run.read_frames()
        _same_recovery(run.read_recovery(), _recovery())
        assert run.flow_fingerprints() == (FINGERPRINT,)
        assert run.read_config().mapping == yaml_copy(_room_raw())
        run.write_frame(0, 0.0, _fields(0), [_budget(1), _budget(2)])
        assert run.read_frames().steps.tolist() == [0]
        RunFile(tmp_path / "none.h5").discard_march()
        assert not (tmp_path / "none.h5").exists()

    def test_no_record_or_a_bad_one_is_refused(self, tmp_path: Path) -> None:
        """Defects caught: an empty record returned when none was written; a
        mapping written as a record."""
        run = RunFile(tmp_path / "run.h5")
        run.write_frame(0, 0.0, _fields(1), [_budget(1), _budget(2)])
        with pytest.raises(KeyError, match="holds no /recovery/"):
            run.read_recovery()
        with pytest.raises(TypeError, match="record must be a RecoveryRecord"):
            run.write_recovery({})  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        ("changed", "error", "match"),
        [
            ({"sensor_names": ("a", 2, "c")}, TypeError, "a sensor name must be a str"),
            ({"held": 1}, TypeError, "held must be a bool"),
            ({"t_final": None}, TypeError, "t_final must be a real number"),
            ({"hold_time": "5"}, TypeError, "hold_time must be a real number"),
            ({"thresholds": np.ones((2, 1))}, ValueError, r"thresholds must be"),
            ({"initial": np.ones(3)}, ValueError, r"initial has shape \(3,\)"),
            ({"recovery_time": np.ones(1)}, ValueError, "recovery_time has shape"),
            ({"defined": np.ones(3, bool)}, ValueError, "defined has shape"),
            ({"mean_time": np.ones(3)}, ValueError, "mean_time has shape"),
            ({"ratio": np.ones(3)}, ValueError, "ratio has shape"),
            ({"local_recovery_max": np.ones(1)}, ValueError, "local_recovery_max has"),
            ({"sensor_time": np.ones((2, 2))}, ValueError, "sensor_time has shape"),
            ({"local_recovery": np.ones((2, 12))}, ValueError, "local_recovery has"),
            ({"local_recovery": np.ones((3, 3, 4))}, ValueError, "local_recovery has"),
        ],
    )
    def test_a_record_whose_parts_disagree_is_refused(
        self, changed: dict, error: type, match: str
    ) -> None:
        """Each shape and type guard of RecoveryRecord, one at a time."""
        with pytest.raises(error, match=match):
            _recovery(**changed)

    def test_the_march_carries_named_arrays(self, tmp_path: Path) -> None:
        """The recovery tracker's state rides in the march checkpoint, bitwise.

        Defects caught: the arrays dropped, or stored as float32; a name that
        is not a str.
        """
        run = RunFile(tmp_path / "run.h5")
        budgets = [_budget(1), _budget(2)]
        carried = {
            "last_above": np.array([[[np.nan, 0.5], [1.0 / 3.0, np.inf]]]),
            "room_since": np.array(0.1 + 0.2),
        }
        run.checkpoint(4, 0.4, _fields(4), budgets, FINGERPRINT, recovery=carried)
        read = run.read_checkpoint().recovery
        assert sorted(read) == sorted(carried)
        for name, array in carried.items():
            assert read[name].dtype == np.float64
            assert read[name].shape == array.shape
            assert read[name].tobytes() == array.tobytes(), name
        run.checkpoint(5, 0.5, _fields(5), budgets, FINGERPRINT)
        assert run.read_checkpoint().recovery == {}
        with pytest.raises(TypeError, match="a recovery name must be a str"):
            run.checkpoint(
                6, 0.6, _fields(6), budgets, FINGERPRINT, recovery={1: np.zeros(1)}
            )


# ---------------------------------------------------------------------------
# Crash safety
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCrashSafety:
    def _run(self, tmp_path: Path) -> RunFile:
        """A run file with two frames, a march checkpoint at step 10 and a steady one."""
        run = RunFile(tmp_path / "run.h5")
        run.write_frame(
            0, 0.0, _fields(1), [_budget(1, fresh=True), _budget(2, fresh=True)]
        )
        run.write_frame(10, 0.5, _fields(2), [_budget(3), _budget(4)])
        run.checkpoint(10, 0.5, _fields(2), [_budget(3), _budget(4)], FINGERPRINT)
        run.write_steady_checkpoint(_steady(True), FINGERPRINT)
        return run

    @pytest.mark.parametrize("writer", ["march", "steady"])
    def test_a_failure_before_the_replace_keeps_the_previous_checkpoints(
        self, tmp_path: Path, writer: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Killed between the temporary file's close and the replace: nothing is lost.

        The march's step 10 and the steady checkpoint read back as written
        before, and the run file's frames read back whole. Defect caught:
        the checkpoint written into the checkpoint file in place (the old
        group deleted, then the new one written), which a kill in between
        leaves holding neither.
        """
        run = self._run(tmp_path)
        before = run.read_frames()

        def killed(*args: object) -> None:
            raise _KilledError

        monkeypatch.setattr("src.io_manager.os.replace", killed)
        with pytest.raises(_KilledError):
            if writer == "march":
                run.checkpoint(
                    20, 1.0, _fields(3), [_budget(5), _budget(6)], FINGERPRINT
                )
            else:
                run.write_steady_checkpoint(_steady(False), OTHER_FINGERPRINT)
        monkeypatch.undo()
        assert run.temporary_path.exists()
        march = run.read_checkpoint()
        assert march.step == 10
        assert march.fields[1].tobytes() == _fields(2)[1].tobytes()
        _same_budgets(march.budgets, (_budget(3), _budget(4)))
        fingerprint, steady = run.read_steady_checkpoint()
        assert fingerprint == FINGERPRINT
        _same_checkpoint(steady, _steady(True))
        after = run.read_frames()
        assert after.steps.tolist() == before.steps.tolist() == [0, 10]
        assert after.fields[0].tobytes() == before.fields[0].tobytes()

    @pytest.mark.parametrize("stale", ["garbage", "hdf5"])
    def test_a_stale_temporary_file_is_overwritten_not_read(
        self, tmp_path: Path, stale: str
    ) -> None:
        """A .tmp an earlier crash left, unreadable or holding other groups, is truncated.

        Defect caught: the temporary file opened for appending, which fails
        on the unreadable one and carries the other's stale group (or its
        checkpoint) into the checkpoint file.
        """
        run = self._run(tmp_path)
        if stale == "garbage":
            run.temporary_path.write_bytes(b"not an HDF5 file")
        else:
            with h5py.File(run.temporary_path, "w") as handle:
                handle.create_group("stale")
                handle.create_group(CHECKPOINT).attrs["step"] = 999
        run.checkpoint(20, 1.0, _fields(3), [_budget(5), _budget(6)], FINGERPRINT)
        assert not run.temporary_path.exists()
        assert run.read_checkpoint().step == 20
        _same_checkpoint(run.read_steady_checkpoint()[1], _steady(True))
        with h5py.File(run.checkpoint_path, "r") as handle:
            assert sorted(handle) == [CHECKPOINT, STEADY_CHECKPOINT]
