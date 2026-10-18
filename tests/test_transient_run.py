"""Tier 2 through ``time_integration.run`` (ADR-014 D and G, step 6b).

A configuration with a ``transient`` section runs the flow through time:
``StaggeredSolver.solve_transient`` from the initial state, the run file
holding the frames, one row per step and the lift and drag, the checkpoint
the two levels BDF2 needs. What is pinned, on the 3 m channel of
tests/test_forces.py with its block at Re 20, 60x20 cells:

- What a new run writes: the step-0 frame (the initial state with the
  boundary's normal velocities), the frames every ``output_interval`` steps
  and at the last, a row for every step whose counts are the solver's and
  whose forces are the gauge's on the step's state.
- The restart: interrupted after a checkpoint and restarted, the run file's
  frames and rows and the end state are the uninterrupted run's to the
  byte; without a checkpoint the restart marches again from the stored
  step-0 frame, to the same bytes; under a later ``t_end`` it continues.
- A step at its cap fails the run with the rows before it written.
- The summary: the drag coefficient's window and the Strouhal refusal.
- scripts/run_scenario.py's tier-2 summary and exit status.
- What cannot be a tier-2 run is refused, at load (tests/test_config.py
  holds the section's own keys) or by ``run``.

The planted defects each test was shown to catch are named in its
docstring (docs/reports/val022/plant68.py).
"""

import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import run_scenario

import src.time_integration as ti
from src.config import SimConfig
from src.forces import ForceGauge, StrouhalReading
from src.io_manager import RunFile
from src.mesh import Mesh
from src.solver_staggered import FlowState, TransientStepError
from src.staggered import allocate_fields
from src.time_integration import Provenance, TransientMarchStep, TransientRunRecord, run
from tests.test_forces import _raw as channel_with_block

DT = 0.05
STEPS = 12
PROVENANCE = Provenance("test", "2026-10-18T00:00:00", "test", 0)


def _raw(steps: int = STEPS, checkpoint: int | None = 4, forces: bool = True) -> dict:
    """The channel with its block under tier 2, Re 20 on the block."""
    raw = channel_with_block(60, 20, freestream=True)
    raw["fluid"]["viscosity"] = 0.01
    raw["solver"].update(dt=DT, t_end=steps * DT, output_interval=5)
    if checkpoint is not None:
        raw["solver"]["checkpoint_interval"] = checkpoint
    raw["transient"] = {"scheme": "bdf2", "step_tol": 1e-6, "step_cap": 300}
    if forces:
        raw["transient"]["forces"] = {"obstacle": "block", "reference_velocity": 1.0}
    return raw


def _config(**kwargs: object) -> SimConfig:
    return SimConfig.from_dict(_raw(**kwargs))  # type: ignore[arg-type]


def _same_files(a: RunFile, b: RunFile) -> None:
    """Two run files hold the same frames and rows, to the byte."""
    fa, fb = a.read_transient_frames(), b.read_transient_frames()
    for name in ("steps", "times", "u", "v", "p"):
        assert np.array_equal(getattr(fa, name), getattr(fb, name)), name
    sa, sb = a.read_step_series(), b.read_step_series()
    for name in (
        "step",
        "outer",
        "pressure_cap_hits",
        "t",
        "worst_imbalance",
        "drag",
        "lift",
        "c_d",
        "c_l",
    ):
        assert np.array_equal(getattr(sa, name), getattr(sb, name), equal_nan=True), (
            name
        )


@pytest.fixture(scope="module")
def whole(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[RunFile, TransientRunRecord, list]:
    """The uninterrupted run, every on_step call kept."""
    path = tmp_path_factory.mktemp("whole") / "run.h5"
    seen: list[TransientMarchStep] = []
    record = run(_config(), path, on_step=seen.append, provenance=PROVENANCE)
    return RunFile(path), record, seen


@pytest.mark.integration
class TestNewRun:
    def test_the_frames_are_step_0_every_output_interval_and_the_last(
        self, whole: tuple
    ) -> None:
        """Steps 0, 5, 10 and 12; step 0 the rest state with the inlet written.

        Catches: the step-0 frame without the boundary's normal velocities,
        the last step's frame left out.
        """
        run_file, record, seen = whole
        frames = run_file.read_transient_frames()
        assert frames.steps.tolist() == [0, 5, 10, 12]
        assert frames.times.tolist() == [0.0, 5 * DT, 10 * DT, 12 * DT]
        mesh = Mesh(_config())
        u0, _v0, _p0 = allocate_fields(mesh)
        assert np.all(frames.u[0][:, 0] == 1.0) and np.all(
            frames.u[0][:, 1:-1] == u0[:, 1:-1]
        )
        for index, step in enumerate(frames.steps[1:], start=1):
            state = seen[step - 1].state
            assert np.array_equal(frames.u[index], state.u)
            assert np.array_equal(frames.p[index], state.p)
        assert record.steps == STEPS and record.t_final == STEPS * DT

    def test_every_step_has_its_row_with_the_solvers_counts_and_the_gauges_forces(
        self, whole: tuple
    ) -> None:
        """Catches: a step's row left out at a frame, the forces of another state."""
        run_file, record, seen = whole
        series = run_file.read_step_series()
        assert series.step.tolist() == list(range(1, STEPS + 1))
        config = _config()
        mesh = Mesh(config)
        gauge = ForceGauge(mesh, config, "block", 1.0)
        for row, step in enumerate(seen):
            assert series.outer[row] == step.record.outer
            assert series.worst_imbalance[row] == step.record.worst_imbalance
            assert series.t[row] == step.record.t
            forces = gauge.forces(step.state.u, step.state.v, step.state.p)
            c_d, c_l = gauge.coefficients(forces)
            assert (series.drag[row], series.lift[row]) == (forces.drag, forces.lift)
            assert (
                (series.c_d[row], series.c_l[row]) == (c_d, c_l) == (step.c_d, step.c_l)
            )
        assert record.outer_mean == pytest.approx(np.mean(series.outer), rel=1e-15)
        assert record.outer_max == int(series.outer.max())

    def test_twelve_steps_give_no_strouhal_number_and_the_last_drag(
        self, whole: tuple
    ) -> None:
        run_file, record, _seen = whole
        series = run_file.read_step_series()
        assert (
            record.strouhal is None and "needs the last 10" in record.strouhal_refusal
        )
        assert (
            record.drag_coefficient == series.c_d[-1] and record.drag_coefficient > 0.5
        )

    def test_the_drag_coefficient_is_the_mean_over_the_strouhal_window(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With a reading whose window starts at step 7, the mean of steps 7 to 12.

        Catches: the mean taken over the whole run.
        """
        reading = StrouhalReading(0.15, 6.0, np.array([7 * DT, 1.0]), 0.15, 0.1, True)
        monkeypatch.setattr(ti, "strouhal", lambda *args: reading)
        record = run(_config(), tmp_path / "run.h5", provenance=PROVENANCE)
        series = RunFile(tmp_path / "run.h5").read_step_series()
        assert record.strouhal is reading
        assert record.drag_coefficient == pytest.approx(
            float(np.mean(series.c_d[6:])), rel=1e-15
        )

    def test_without_forces_the_rows_carry_nan_and_no_reading(self, tmp_path) -> None:
        record = run(
            _config(forces=False, checkpoint=None),
            tmp_path / "run.h5",
            provenance=PROVENANCE,
        )
        series = RunFile(tmp_path / "run.h5").read_step_series()
        assert np.all(np.isnan(series.c_l)) and np.all(np.isnan(series.drag))
        assert (
            record.drag_coefficient is None
            and record.strouhal_refusal == "no transient.forces configured"
        )

    def test_an_initial_state_is_the_step_0_frame(self, tmp_path) -> None:
        config = _config()
        u, v, p = allocate_fields(Mesh(config))
        u[:, 1:-1] = 0.5
        run(
            config,
            tmp_path / "run.h5",
            provenance=PROVENANCE,
            initial=FlowState(u=u, v=v, p=p),
        )
        frames = RunFile(tmp_path / "run.h5").read_transient_frames()
        assert np.all(frames.u[0][:, 1:-1] == 0.5) and np.all(frames.u[0][:, 0] == 1.0)


class _InterruptError(Exception):
    """The interruption a crash would be."""


def _interrupted(path, at: int, **kwargs: object) -> None:
    def crash(step: TransientMarchStep) -> None:
        if step.record.step == at:
            raise _InterruptError

    with pytest.raises(_InterruptError):
        run(_config(**kwargs), path, on_step=crash, provenance=PROVENANCE)  # type: ignore[arg-type]


@pytest.mark.integration
class TestRestart:
    def test_a_restart_from_the_checkpoint_is_the_uninterrupted_run(
        self, tmp_path, whole: tuple
    ) -> None:
        """Interrupted at step 10 (checkpoint at 8, rows to 10 written): restarted.

        Catches: the previous level not checkpointed (BDF2 resumed as
        implicit Euler), the rows after the checkpoint not discarded, the
        checkpoint taken before the step's rows.
        """
        path = tmp_path / "run.h5"
        _interrupted(path, 10)
        assert RunFile(path).read_step_series().step[-1] == 10
        record = run(_config(), restart=path)
        assert record.restarted_from == 8 and record.steps == STEPS
        _same_files(RunFile(path), whole[0])

    def test_a_restart_without_a_checkpoint_marches_again_from_step_0(
        self, tmp_path
    ) -> None:
        """No checkpoint_interval: the stored step-0 frame starts it again."""
        first, second = tmp_path / "a.h5", tmp_path / "b.h5"
        run(_config(checkpoint=None), first, provenance=PROVENANCE)
        _interrupted(second, 7, checkpoint=None)
        record = run(_config(checkpoint=None), restart=second)
        assert record.restarted_from == 0
        _same_files(RunFile(second), RunFile(first))

    def test_a_later_t_end_continues_the_run(self, tmp_path) -> None:
        """Twelve steps, then sixteen by restart, against sixteen at once.

        The rows are the same to the byte; the frames too, but for the
        twelve-step run's last frame, which the extended file keeps.
        """
        path, longer = tmp_path / "run.h5", tmp_path / "longer.h5"
        run(_config(), path, provenance=PROVENANCE)
        record = run(_config(steps=16), restart=path)
        run(_config(steps=16), longer, provenance=PROVENANCE)
        assert record.restarted_from == STEPS and record.steps == 16
        extended, once = (
            RunFile(path).read_transient_frames(),
            RunFile(longer).read_transient_frames(),
        )
        assert extended.steps.tolist() == [0, 5, 10, 12, 15, 16]
        assert once.steps.tolist() == [0, 5, 10, 15, 16]
        keep = np.isin(extended.steps, once.steps)
        for name in ("times", "u", "v", "p"):
            assert np.array_equal(getattr(extended, name)[keep], getattr(once, name)), (
                name
            )
        sa, sb = RunFile(path).read_step_series(), RunFile(longer).read_step_series()
        for name in (
            "step",
            "outer",
            "t",
            "worst_imbalance",
            "drag",
            "lift",
            "c_d",
            "c_l",
        ):
            assert np.array_equal(getattr(sa, name), getattr(sb, name)), name

    def test_a_restart_at_or_past_t_end_or_under_another_configuration_is_refused(
        self, tmp_path
    ) -> None:
        path = tmp_path / "run.h5"
        run(_config(), path, provenance=PROVENANCE)
        with pytest.raises(ValueError, match=r"at or past solver\.t_end"):
            run(_config(), restart=path)
        other = _raw()
        other["transient"]["step_tol"] = 1e-5
        with pytest.raises(ValueError, match="step_tol"):
            run(SimConfig.from_dict(other), restart=path)


@pytest.mark.integration
def test_a_step_at_its_cap_fails_with_the_rows_before_it(tmp_path) -> None:
    """step_cap 1 cannot meet 1e-6: the error, and no row past the steps that met it."""
    raw = _raw()
    raw["transient"]["step_cap"] = 1
    path = tmp_path / "run.h5"
    with pytest.raises(TransientStepError):
        run(SimConfig.from_dict(raw), path, provenance=PROVENANCE)
    run_file = RunFile(path)
    assert run_file.read_transient_frames().steps.tolist() == [0]
    with pytest.raises(KeyError):
        run_file.read_step_series()


@pytest.mark.unit
class TestRefusals:
    def test_what_run_cannot_take_under_tier_2_is_refused(self, tmp_path) -> None:
        config = _config()
        rest = FlowState(*allocate_fields(Mesh(config)))
        with pytest.raises(ValueError, match="flows_from"):
            run(config, tmp_path / "a.h5", flows_from=tmp_path / "b.h5")
        with pytest.raises(ValueError, match="exactly one"):
            run(config)
        with pytest.raises(ValueError, match="initial is a new run"):
            run(config, restart=tmp_path / "a.h5", initial=rest)

    def test_neither_section_is_refused(self, tmp_path) -> None:
        raw = _raw()
        del raw["transient"]
        with pytest.raises(ValueError, match="no transient section"):
            run(SimConfig.from_dict(raw), tmp_path / "a.h5")

    def test_an_existing_run_file_is_not_overwritten(self, tmp_path) -> None:
        path = tmp_path / "run.h5"
        path.write_bytes(b"")
        with pytest.raises(ValueError, match="exists"):
            run(_config(), path, provenance=PROVENANCE)

    def test_the_transient_section_refuses_what_it_cannot_run(self) -> None:
        """Turbulence, a fraction of a step, an unknown obstacle, unknown keys."""
        cases = []
        raw = _raw()
        raw["solver"]["t_end"] = 12.5 * DT
        cases.append((raw, "whole number of steps"))
        raw = _raw()
        raw["transient"]["forces"]["obstacle"] = "missing"
        cases.append((raw, "not an obstacle"))
        raw = _raw()
        raw["transient"]["forces"]["area"] = 1.0
        cases.append((raw, "exactly the keys"))
        raw = _raw()
        raw["transient"]["sweeps"] = 3
        cases.append((raw, "not recognised"))
        raw = _raw()
        raw["transient"]["scheme"] = "crank_nicolson"
        cases.append((raw, "scheme must be one of"))
        for given, match in cases:
            with pytest.raises(ValueError, match=match):
                SimConfig.from_dict(copy.deepcopy(given))


@pytest.mark.integration
class TestRunScenarioSummary:
    """scripts/run_scenario.py on a transient file: the run and its summary."""

    def test_a_transient_file_runs_and_prints_its_summary(
        self, tmp_path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        source = tmp_path / "transient.yaml"
        source.write_text(yaml.safe_dump(_raw(checkpoint=None)), encoding="utf-8")
        status = run_scenario.main(
            [str(source), "--out", str(tmp_path / "run.h5"), "--quiet"]
        )
        lines = capsys.readouterr().out.splitlines()
        series = RunFile(tmp_path / "run.h5").read_step_series()
        assert status == run_scenario.EXIT_DONE
        assert lines[2] == f"steps: {STEPS} to t = {STEPS * DT:.6g} s"
        assert lines[3] == (
            f"outer per step: mean {np.mean(series.outer):.2f}, largest "
            f"{int(series.outer.max())}"
        )
        assert lines[4].startswith("St: not defined (the lift holds")
        assert lines[5] == f"C_D: {series.c_d[-1]:.5f} (at the last step)"

    def test_a_reading_is_printed_with_its_cross_check(self, tmp_path) -> None:
        reading = StrouhalReading(0.14612, 6.8437, np.zeros(11), 0.1459, 0.0021, True)
        record = TransientRunRecord(
            path=tmp_path / "run.h5",
            steps=4000,
            t_final=200.0,
            wall_seconds=12.0,
            outer_mean=5.25,
            outer_max=128,
            strouhal=reading,
            strouhal_refusal=None,
            drag_coefficient=1.4723,
            restarted_from=2000,
        )
        lines = run_scenario.transient_summary(record, tmp_path / "f.yaml")
        assert lines[2] == "restarted from step 2000"
        assert lines[5] == (
            "St: 0.14612 over the last ten periods (period 6.84370 s); spectrum "
            "0.14590 +/- 0.00210, agrees"
        )
        assert lines[6] == "C_D: 1.47230 (over the St window)"

    def test_a_step_at_its_cap_exits_as_not_converged(self, tmp_path) -> None:
        raw = _raw(checkpoint=None)
        raw["transient"]["step_cap"] = 1
        source = tmp_path / "transient.yaml"
        source.write_text(yaml.safe_dump(raw), encoding="utf-8")
        status = run_scenario.main(
            [str(source), "--out", str(tmp_path / "run.h5"), "--quiet"]
        )
        assert status == run_scenario.EXIT_NOT_CONVERGED
