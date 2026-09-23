"""Unit tests for the field viewer in scripts/view_field.py."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import view_field  # noqa: E402 -- scripts/ is not a package; path set above


@pytest.mark.unit
def test_render_refuses_an_unrecognised_case_kind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A kind that is neither cavity nor poiseuille raises instead of drawing the parabola.

    load_case is stood in for so the unknown kind reaches the panel dispatch,
    which is what happens once a third case family has a configuration file.
    """
    real_load_case = view_field.load_case
    monkeypatch.setattr(
        view_field,
        "load_case",
        lambda kind, grid=None: real_load_case("cavity", grid=grid),
    )
    n = 6
    ramp = np.tile(np.linspace(0.0, 1.0, n), (n, 1))
    npz = tmp_path / "clean_room_6x6.npz"
    np.savez(
        npz,
        kind=np.array("clean_room"),
        case_id=np.array("clean_room_6x6"),
        nx=n,
        ny=n,
        u=ramp,
        v=np.zeros((n, n)),
        p=ramp.T,
    )

    with pytest.raises(ValueError, match="clean_room"):
        view_field.render(npz, tmp_path)
    view_field.plt.close("all")
    assert not list(tmp_path.glob("*.png"))


@pytest.mark.unit
def test_render_draws_and_scores_a_cavity_on_the_true_centerlines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cavity panel takes its profiles and its score from the true centerlines.

    Both functions are wrapped to record their calls. A viewer that went back
    to the offset sampling for either would not call it.
    """
    calls: list[str] = []

    def spy(name: str) -> None:
        real = getattr(view_field, name)

        def wrapped(*args: object) -> object:
            calls.append(name)
            return real(*args)

        monkeypatch.setattr(view_field, name, wrapped)

    spy("cavity_true_centerline_profiles")
    spy("cavity_true_centerline_errors")
    n = 8
    ramp = np.tile(np.linspace(0.0, 1.0, n), (n, 1))
    npz = tmp_path / "val002_8x8.npz"
    np.savez(
        npz,
        kind=np.array("cavity"),
        case_id=np.array("val002_8x8"),
        nx=n,
        ny=n,
        u=0.3 + 0.8 * ramp,
        v=0.3 + 0.8 * ramp.T,
        p=ramp.T,
        outer_iterations=np.array(1),
    )

    png = view_field.render(npz, tmp_path)
    view_field.plt.close("all")
    assert png.exists()
    assert calls == ["cavity_true_centerline_profiles", "cavity_true_centerline_errors"]
