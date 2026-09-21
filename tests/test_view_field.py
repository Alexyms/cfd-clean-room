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
