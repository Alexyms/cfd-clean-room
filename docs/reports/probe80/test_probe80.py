"""Prompt 80, section 3: the structural check on the real cumulative maps.

A channel's cumulative map is never later than its class's map in any cell, and the
largest class's two maps are identical (no larger class adds to it). Run on prompt 79's
five 320x120 runs and this pass's three 80x30 probe runs, from their stored frames, and on
the probe runs' step maps. The run files are untracked; a case without its file skips.

    python -m pytest docs/reports/probe80/test_probe80.py
    PROBE80_PLANT=smaller python -m pytest docs/reports/probe80/test_probe80.py

The second line plants the defect (the cumulative channel summed over class k and the
smaller classes), which the frame cases must catch. The step maps were made in the probe
runs' own processes, without the plant, so the plant does not reach them.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import probe80  # noqa: E402 -- follows sys.path.insert

from src.config import SimConfig  # noqa: E402
from src.io_manager import RunFile  # noqa: E402
from src.mesh import SOLID, Mesh  # noqa: E402
from src.monitor import contaminating_events  # noqa: E402

FINE = ROOT / "results" / "builder79" / "product"
COARSE = ROOT / "results" / "builder80" / "placement"
FRAME_RUNS = [
    *(FINE / f"{n}.h5" for n in ("door_leak", "filter_breach", "equipment_dust")),
    *(FINE / f"{n}.h5" for n in ("seal_leak", "clean_out")),
    *(COARSE / f"{n}.h5" for n in ("door", "breach", "dust")),
]
STEP_RUNS = [COARSE / f"{n}.h5" for n in ("door", "breach", "dust")]


def inputs(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The thresholds, the live cells and the onsets a run file records."""
    config = SimConfig.from_dict(RunFile(path).read_config().mapping)
    live = Mesh(config).cell_type != SOLID
    onsets = np.array([o for _, o in contaminating_events(config.scenario)])
    return np.array(config.class_thresholds("prompt 80")), live, onsets


def assert_structure(per_class: np.ndarray, cumulative: np.ndarray) -> None:
    """Never later in any live cell; the largest channel identical; SOLID alike."""
    live = ~np.isnan(per_class)
    assert np.array_equal(live, ~np.isnan(cumulative)), "SOLID cells differ"
    later = cumulative[live] > per_class[live]
    assert not later.any(), f"{int(later.sum())} entries are later cumulatively"
    assert per_class[:, -1].tobytes() == cumulative[:, -1].tobytes(), (
        "the largest channel's cumulative map is not its class's"
    )


@pytest.mark.integration
@pytest.mark.parametrize("path", FRAME_RUNS, ids=lambda p: f"{p.parent.name}-{p.stem}")
def test_the_cumulative_frame_maps_are_never_later(path: Path) -> None:
    """From the stored frames: never later anywhere, the largest class identical."""
    if not path.exists():
        pytest.skip(f"{path} is not on this machine")
    thresholds, live, onsets = inputs(path)
    maps, _, _ = probe80.frame_maps(path, thresholds, live, onsets)
    assert_structure(maps[probe80.PER_CLASS], maps[probe80.CUMULATIVE])


@pytest.mark.integration
@pytest.mark.parametrize("path", STEP_RUNS, ids=lambda p: f"steps-{p.stem}")
def test_the_cumulative_step_maps_are_never_later(path: Path) -> None:
    """From the probe run's every-step maps: the same, and per-class equal to the march's."""
    saved_path = path.with_name(f"{path.stem}.channels.npz")
    if not saved_path.exists():
        pytest.skip(f"{saved_path} is not on this machine")
    saved = np.load(saved_path)
    _, live, _ = inputs(path)
    per_class, cumulative = saved["per_class"], saved["cumulative"]
    assert per_class.tobytes() == saved["own"].tobytes()
    # NaN until reached in the trackers; inf for a live cell never reached, as the maps.
    for array in (per_class, cumulative):
        array[np.isnan(array) & live] = np.inf
    assert_structure(per_class, cumulative)


@pytest.mark.unit
def test_the_channels_sum_the_larger_classes() -> None:
    """On planted fields, channel k is class k and every larger class; per-class is k alone."""
    fields = {k: np.full((1, 2), 10.0**k) for k in range(5)}
    assert probe80.channel(fields, 4, probe80.CUMULATIVE)[0, 0] == 10_000.0
    assert probe80.channel(fields, 2, probe80.CUMULATIVE)[0, 0] == 11_100.0
    assert probe80.channel(fields, 2, probe80.PER_CLASS)[0, 0] == 100.0
