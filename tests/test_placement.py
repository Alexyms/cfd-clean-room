"""The arrival maps' readers (src/placement.py, ADR-015 D; step 3).

ADR-015 H's step 3 builds only ``arrival_from_frames`` and
``arrival_from_run``; the scoring, the candidates and the search are step
4's. What is pinned here, unit tier, each test naming in its docstring the
planted defect it was shown to catch:

- ``arrival_from_frames`` on planted frames: the first frame time at or
  after each onset at or above the threshold (a frame at the onset counts),
  inf where none is, NaN in SOLID cells, a second event's map from its own
  onset, each class against its own threshold; its refusals.
- ``arrival_from_run``'s refusals on files it cannot read: a recorded
  configuration with no scenario section, and a file with neither maps nor
  frames.

The march's maps against the frames' on a coarse run, and a Phase 4 file
read through ``arrival_from_run`` and flagged, are integration tests in
tests/test_time_integration.py.
"""

from pathlib import Path

import numpy as np
import pytest
import yaml

from src.io_manager import RunFile, TransportFrames
from src.placement import arrival_from_frames, arrival_from_run
from src.solver_transport import MassBudget

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
# A 2x3 grid with one SOLID cell and two classes, frames at 0, 0.5, 1 and 2 s.
LIVE = np.array([[True, True, False], [True, True, True]])
TIMES = np.array([0.0, 0.5, 1.0, 2.0])
THRESHOLDS = np.array([1.0, 10.0])
PROVENANCE = {
    "commit": "test",
    "started": "2026-11-01T00:00:00",
    "machine": "test-machine",
    "concurrent": 0,
}


def frames_of(cells: dict[tuple[int, int], list[float]]) -> TransportFrames:
    """Four frames of two classes: CELLS' class-0 series, ten times it in class 1."""
    stack = np.zeros((TIMES.size, *LIVE.shape))
    for (j, i), series in cells.items():
        stack[:, j, i] = series
    budgets = tuple((MassBudget(), MassBudget()) for _ in TIMES)
    return TransportFrames(
        steps=np.array([0, 50, 100, 200]),
        times=TIMES,
        fields={0: stack, 1: 10.0 * stack},
        budgets=budgets,
    )


@pytest.mark.unit
class TestArrivalFromFrames:
    def test_the_first_frame_at_or_after_each_onset_at_or_above(self) -> None:
        """Onsets 0 and 0.75: per cell the first such frame, inf where none, NaN SOLID.

        Cell (0, 0) is above from 0.5 s: event 0 reads 0.5, event 1 the
        first frame after its onset, 1.0. Cell (0, 1) is exactly at the
        threshold at 0.5 only: event 0 reads 0.5 and event 1 never. Cell
        (1, 2) is above at 2.0: both read 2.0. Defects caught: ">" for "at
        or above" ((0, 1) inf); the second event's map started at t = 0
        ((0, 0) 0.5 for event 1); the last frame kept ((0, 0) 2.0); the
        SOLID cell given inf.
        """
        frames = frames_of(
            {
                (0, 0): [0.0, 2.0, 2.0, 2.0],
                (0, 1): [0.0, 1.0, 0.0, 0.0],
                (1, 2): [0.0, 0.0, 0.0, 5.0],
                (0, 2): [9.0, 9.0, 9.0, 9.0],
            }
        )
        maps = arrival_from_frames(frames, THRESHOLDS, LIVE, np.array([0.0, 0.75]))
        assert maps.shape == (2, 2, 2, 3)
        for k in (0, 1):
            assert maps[0, k].tolist()[0][:2] == [0.5, 0.5]
            assert maps[1, k].tolist()[0][:2] == [1.0, np.inf]
            assert maps[0, k, 1, 2] == maps[1, k, 1, 2] == 2.0
            assert np.isinf(maps[:, k, 1, :2]).all()
        assert np.isnan(maps[:, :, 0, 2]).all()

    def test_a_frame_at_an_onset_counts_for_that_event(self) -> None:
        """A frame whose time is an onset is the first frame at or after it.

        Onsets 0 and 1 s, both frame times. Cell (0, 0) is above in the
        frame at 0 only: event 0 reads 0.0 and event 1 never. Cell (0, 1) is
        above in the frame at 1.0 only: both events read 1.0. Every clean-out
        has this tie (its onset and its first frame are both t = 0). Defect
        caught: ">" for ">=" in the comparison with the onset ((0, 0) inf
        for event 0, (0, 1) inf for event 1).
        """
        frames = frames_of({(0, 0): [2.0, 0.0, 0.0, 0.0], (0, 1): [0.0, 0.0, 2.0, 0.0]})
        maps = arrival_from_frames(frames, THRESHOLDS, LIVE, np.array([0.0, 1.0]))
        for k in (0, 1):
            assert maps[0, k, 0, 0] == 0.0 and np.isinf(maps[1, k, 0, 0])
            assert maps[0, k, 0, 1] == maps[1, k, 0, 1] == 1.0

    def test_a_class_with_its_own_threshold(self) -> None:
        """Each class against its own threshold, not class 0's and not the last.

        Class 1's field is ten times class 0's and its threshold twenty
        times, so the cell reaches class 0's threshold at 1.0 s and never
        reaches class 1's. Defects caught: class 0's threshold used for every
        class (class 1 reads 0.5); the last class's used for every class
        (class 0 never).
        """
        frames = frames_of({(1, 0): [0.0, 0.5, 1.0, 1.0]})
        maps = arrival_from_frames(frames, np.array([1.0, 20.0]), LIVE, np.array([0.0]))
        assert maps[0, 0, 1, 0] == 1.0 and np.isinf(maps[0, 1, 1, 0])

    @pytest.mark.parametrize(
        ("thresholds", "live", "onsets", "match"),
        [
            (np.array([1.0]), LIVE, np.array([0.0]), r"thresholds of shape \(1,\)"),
            (THRESHOLDS[None], LIVE, np.array([0.0]), r"thresholds of shape \(1, 2\)"),
            (THRESHOLDS, LIVE[:, :2], np.array([0.0]), "frames are"),
            (THRESHOLDS, LIVE, np.array([[0.0]]), r"onsets must be \[events\]"),
        ],
    )
    def test_the_inputs_are_refused_unless_they_fit_the_frames(
        self, thresholds: np.ndarray, live: np.ndarray, onsets: np.ndarray, match: str
    ) -> None:
        """One threshold per class of the frames, live of their shape, onsets [events]."""
        with pytest.raises(ValueError, match=match):
            arrival_from_frames(frames_of({}), thresholds, live, onsets)

    def test_frames_whose_classes_are_not_0_to_n_minus_1_are_refused(self) -> None:
        """A frames object holding classes 0 and 2 has no class 1 to read."""
        frames = frames_of({})
        frames.fields[2] = frames.fields.pop(1)
        with pytest.raises(ValueError, match=r"frames' classes \[0, 2\]"):
            arrival_from_frames(frames, THRESHOLDS, LIVE, np.array([0.0]))


@pytest.mark.unit
class TestArrivalFromRun:
    def test_a_file_recording_no_scenario_is_refused(self, tmp_path: Path) -> None:
        """A tier-2 or bare configuration has no plan, so no events to map."""
        raw = yaml.safe_load(
            (CONFIGS / "clean_room_default.yaml").read_text(encoding="utf-8")
        )
        raw["domain"].update({"nx": 80, "ny": 30})
        run = RunFile(tmp_path / "run.h5")
        run.write_config(raw, **PROVENANCE)
        with pytest.raises(ValueError, match="records no scenario section"):
            arrival_from_run(run.path)

    def test_a_file_with_neither_maps_nor_frames_is_refused(
        self, tmp_path: Path
    ) -> None:
        """A run killed before its first frame: KeyError from the frames' reader."""
        raw = yaml.safe_load(
            (CONFIGS / "scenario_door_leak.yaml").read_text(encoding="utf-8")
        )
        raw["scenario"]["base"] = str(CONFIGS / "clean_room_default.yaml")
        raw["domain"] = {"nx": 80, "ny": 30}
        run = RunFile(tmp_path / "run.h5")
        run.write_config(raw, **PROVENANCE)
        with pytest.raises(KeyError, match="holds no /transport/ frames"):
            arrival_from_run(run)
