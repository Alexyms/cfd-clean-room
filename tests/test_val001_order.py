"""Unit tests for the VAL-001 order study in scripts/val001_order.py.

The control is the study's credibility, so it is tested both ways: it
recovers the orders its synthetic fields carry, and it stops on the two
pipeline defects it exists to catch.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import val001_order  # noqa: E402 -- scripts/ is not a package; path set above


@pytest.mark.unit
def test_control_recovers_orders_two_and_one() -> None:
    """All six orders, reference-free and against the parabola, within 0.05 of q."""
    results = val001_order.run_control(0.1)
    assert set(results) == {"2.0", "1.0"}
    for q, found in results.items():
        orders = [p for values in found.values() for p in values]
        assert len(orders) == 6
        assert all(abs(p - float(q)) < val001_order.CONTROL_TOL for p in orders)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("name", "defect"),
    [
        ("restrict", lambda f: f[::2]),
        ("station", lambda u, fraction: u[:, u.shape[1] // 2]),
    ],
    ids=["every-other-cell", "column-nx-over-2"],
)
def test_control_stops_on_a_pipeline_defect(
    monkeypatch: pytest.MonkeyPatch, name: str, defect: object
) -> None:
    """Restriction by sampling, or one column instead of the face, stops the script."""
    monkeypatch.setattr(val001_order, name, defect)
    with pytest.raises(SystemExit, match="control failed"):
        val001_order.run_control(0.1)


@pytest.mark.unit
def test_station_refuses_a_position_that_is_not_a_face() -> None:
    """3L/4 of ten columns falls inside a cell, where no two columns straddle it."""
    with pytest.raises(ValueError, match="not a face"):
        val001_order.station(np.zeros((4, 10)), 0.75)


@pytest.mark.unit
def test_saved_solve_is_reused_only_while_its_parameters_match(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A matching file returns without a solver; one parameter changed builds one."""

    class SolverBuiltError(Exception):
        pass

    def no_solver(*args: object) -> None:
        raise SolverBuiltError

    monkeypatch.setattr(val001_order, "FIELD_DIR", tmp_path)
    monkeypatch.setattr(val001_order, "StaggeredSolver", no_solver)
    config = val001_order.load_case("poiseuille", grid=(12, 6))
    params = val001_order.json.dumps(
        val001_order.solver_parameters(config), sort_keys=True
    )
    path = tmp_path / "poiseuille_12x6.npz"
    np.savez(path, params=params, u=np.ones((6, 12)))
    assert np.array_equal(val001_order.solve(12, 6)["u"], np.ones((6, 12)))
    np.savez(path, params=params.replace("1e-06", "1e-07"), u=np.ones((6, 12)))
    with pytest.raises(SolverBuiltError):
        val001_order.solve(12, 6)
