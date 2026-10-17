"""The run file's steady checkpoint (src/io_manager.py, ADR-014 D, step 0).

Step 0 builds only ``/steady_checkpoint/``: a checkpoint written and read
back is the checkpoint, bitwise, and a solve resumed from what was read
reaches the uninterrupted solve bitwise. Step 3 adds the rest of the run
file, and its tests (the march's restart, the configuration hash) join
these. Each test names the planted defect it was shown to catch.
"""

from pathlib import Path

import h5py
import numpy as np
import pytest

from src.io_manager import (
    STEADY_CHECKPOINT,
    read_steady_checkpoint,
    write_steady_checkpoint,
)
from src.solver_staggered import FlowState, SteadyCheckpoint
from tests.test_warm_start import FIELDS, FIXTURES, _same_bits, _solved


def _same_checkpoint(a: SteadyCheckpoint, b: SteadyCheckpoint) -> None:
    """Every array, series and count of two checkpoints, bit for bit."""
    for name in FIELDS:
        x, y = getattr(a.state, name), getattr(b.state, name)
        assert (x is None) == (y is None), name
        if x is not None:
            assert x.dtype == y.dtype == np.float64
            assert x.shape == y.shape and x.tobytes() == y.tobytes(), name
    assert (a.state.stop_reason, a.state.outer) == (b.state.stop_reason, b.state.outer)
    for name in ("residual_history", "velocity_steps", "viscosity_steps"):
        assert getattr(a, name) == getattr(b, name), name
    assert (a.rule_version, a.pressure_cap_hits) == (
        b.rule_version,
        b.pressure_cap_hits,
    )


@pytest.mark.integration
@pytest.mark.parametrize("name", list(FIXTURES))
def test_a_checkpoint_read_back_resumes_to_the_uninterrupted_solve(
    name: str, tmp_path: Path
) -> None:
    """Cut at 150, written, read back and resumed: the uninterrupted solve, bitwise.

    Defects caught: a field written as float32; the rule's window or the
    residual history not written; k, eps or nu_t dropped from a coupled
    checkpoint; the outer count not read back.
    """
    make = FIXTURES[name]
    checkpoint = _solved(make(cap=150)).steady_checkpoint
    path = tmp_path / "run.h5"
    write_steady_checkpoint(path, checkpoint)
    read = read_steady_checkpoint(path)
    _same_checkpoint(read, checkpoint)
    _same_bits(_solved(make(), resume=read), _solved(make()))


@pytest.mark.unit
class TestTheGroup:
    def _checkpoint(
        self, turbulent: bool, reason: str | None = "max_simple_iter"
    ) -> SteadyCheckpoint:
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
            state, (0.5, 0.25), (3e-7, 2e-7), window, 5 if turbulent else None, 1
        )

    @pytest.mark.parametrize("turbulent", [False, True])
    @pytest.mark.parametrize("reason", ["max_simple_iter", None])
    def test_what_is_written_is_what_is_read(
        self, tmp_path: Path, turbulent: bool, reason: str | None
    ) -> None:
        """Defects caught: a None stop or rule version read back as a string or 0."""
        checkpoint = self._checkpoint(turbulent, reason)
        write_steady_checkpoint(tmp_path / "run.h5", checkpoint)
        _same_checkpoint(read_steady_checkpoint(tmp_path / "run.h5"), checkpoint)

    def test_a_rewrite_replaces_the_group_and_keeps_the_others(
        self, tmp_path: Path
    ) -> None:
        """Defects caught: the file truncated; a second write refused or merged.

        The coupled checkpoint is written over the laminar one, then the
        laminar one over it, so k, eps and nu_t must go as well as come.
        """
        path = tmp_path / "run.h5"
        with h5py.File(path, "w") as run_file:
            run_file.create_dataset("config", data=b"kept")
        laminar, coupled = self._checkpoint(False), self._checkpoint(True)
        write_steady_checkpoint(path, coupled)
        write_steady_checkpoint(path, laminar)
        _same_checkpoint(read_steady_checkpoint(path), laminar)
        with h5py.File(path, "r") as run_file:
            assert run_file["config"][()] == b"kept"
            assert "k" not in run_file[STEADY_CHECKPOINT]

    def test_a_file_without_the_group_is_refused(self, tmp_path: Path) -> None:
        """Defect caught: an empty checkpoint returned for a file that holds none."""
        path = tmp_path / "run.h5"
        with h5py.File(path, "w"):
            pass
        with pytest.raises(KeyError, match="holds no /steady_checkpoint/ group"):
            read_steady_checkpoint(path)

    def test_what_is_not_a_checkpoint_is_refused(self, tmp_path: Path) -> None:
        """Defect caught: a FlowState written as if it were a checkpoint."""
        with pytest.raises(TypeError, match="checkpoint must be a SteadyCheckpoint"):
            state = self._checkpoint(False).state
            write_steady_checkpoint(tmp_path / "run.h5", state)  # type: ignore[arg-type]
