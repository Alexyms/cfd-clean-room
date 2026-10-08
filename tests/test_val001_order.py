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
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import val001_order  # noqa: E402 -- scripts/ is not a package; path set above

from src.config import SimConfig  # noqa: E402 -- follows sys.path.insert
from validation.cases import (  # noqa: E402 -- follows sys.path.insert
    case_path,
    load_wall_clustered,
)
from validation.metrics import (  # noqa: E402 -- follows sys.path.insert
    poiseuille_reference,
)


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
@pytest.mark.parametrize("fraction", [0.0, 1.0])
def test_station_refuses_the_walls_instead_of_wrapping(fraction: float) -> None:
    """x = 0 would read column -1 and column 0, a mean of the two ends (review 25 S7).

    Both positions are faces of ten columns, so only the interior check
    refuses them.
    """
    u = np.arange(40.0).reshape(4, 10)
    with pytest.raises(ValueError, match="interior face"):
        val001_order.station(u, fraction)
    # An interior face still reads the mean of its two columns.
    assert np.array_equal(val001_order.station(u, 0.5), 0.5 * (u[:, 4] + u[:, 5]))


@pytest.mark.unit
def test_station_refuses_a_position_that_is_not_a_face() -> None:
    """3L/4 of ten columns falls inside a cell, where no two columns straddle it."""
    with pytest.raises(ValueError, match="not a face"):
        val001_order.station(np.zeros((4, 10)), 0.75)


def _channel(grid: tuple[int, int] = (12, 6), **changes: float) -> SimConfig:
    """The channel case at a grid with domain, fluid or inlet values replaced."""
    raw = yaml.safe_load(case_path("poiseuille").read_text(encoding="utf-8"))
    raw["domain"]["nx"], raw["domain"]["ny"] = grid
    raw["domain"].update({k: v for k, v in changes.items() if k in ("width", "height")})
    raw["fluid"].update({k: v for k, v in changes.items() if k == "viscosity"})
    if "velocity" in changes:
        raw["boundaries"]["inlet"]["velocity"] = changes["velocity"]
    return SimConfig.from_dict(raw)


@pytest.mark.unit
@pytest.mark.parametrize(
    "changed",
    [
        {"width": 5.0},
        {"height": 0.7},
        {"viscosity": 2.0e-3},
        {"velocity": 0.05},
        {"grid": (12, 8)},
    ],
    ids=["width", "height", "viscosity", "inlet", "grid"],
)
def test_reuse_key_covers_the_case(changed: dict) -> None:
    """A saved field is reused only for the same case, not only the same solver block.

    Review 25 S8: the key held the solver parameters and the two versions, so
    editing the domain, the fluid or the inlet in the case file left a field
    solved under the old values to be served for the new ones.
    """
    base = val001_order.reuse_key(_channel())
    assert val001_order.reuse_key(_channel()) == base
    assert val001_order.reuse_key(_channel(**changed)) != base


@pytest.mark.unit
def test_reuse_key_covers_the_mesh() -> None:
    """The wall-clustered mesh and the uniform one of the same size key apart."""
    uniform = val001_order.reuse_key(_channel())
    clustered = val001_order.reuse_key(load_wall_clustered("poiseuille", grid=(12, 6)))
    assert clustered != uniform


@pytest.mark.unit
def test_a_solve_that_reaches_its_cap_stops_the_script(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two outer iterations on 12x6 do not converge: SystemExit, nothing saved (review 25 T1)."""
    real = val001_order.load_case

    def capped(name: str, grid: tuple[int, int] | None = None) -> SimConfig:
        config = real(name, grid=grid)
        config.max_simple_iter = 2
        return config

    monkeypatch.setattr(val001_order, "FIELD_DIR", tmp_path)
    monkeypatch.setattr(val001_order, "load_case", capped)
    with pytest.raises(SystemExit, match=r"12x6 stopped at its cap, 2"):
        val001_order.solve(12, 6)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.unit
def test_the_developed_profile_is_the_metrics_analytic_one() -> None:
    """Review 25 S8: one copy of the parabola, in validation.metrics, on the unit height."""
    s = (np.arange(8) + 0.5) / 8
    got = val001_order.developed(8, 0.1)
    assert np.array_equal(got, poiseuille_reference(s, 1.0, 0.1))
    assert got.max() < 1.5 * 0.1
    assert not hasattr(val001_order, "parabola")


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
    params = val001_order.reuse_key(config)
    path = tmp_path / "poiseuille_12x6.npz"
    np.savez(path, params=params, u=np.ones((6, 12)))
    assert np.array_equal(val001_order.solve(12, 6)["u"], np.ones((6, 12)))
    np.savez(path, params=params.replace("1e-06", "1e-07"), u=np.ones((6, 12)))
    with pytest.raises(SolverBuiltError):
        val001_order.solve(12, 6)


@pytest.mark.unit
def test_saved_solve_is_not_reused_under_another_rule_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A file saved under one RULE_VERSION builds a solver under another.

    Defect caught: the version dropped from the key. A new stopping condition
    changes no solver parameter, so without it the file would be reused.
    """

    class SolverBuiltError(Exception):
        pass

    def no_solver(*args: object) -> None:
        raise SolverBuiltError

    monkeypatch.setattr(val001_order, "FIELD_DIR", tmp_path)
    monkeypatch.setattr(val001_order, "StaggeredSolver", no_solver)
    config = val001_order.load_case("poiseuille", grid=(12, 6))
    params = val001_order.reuse_key(config)
    np.savez(tmp_path / "poiseuille_12x6.npz", params=params, u=np.ones((6, 12)))
    monkeypatch.setattr(val001_order, "RULE_VERSION", val001_order.RULE_VERSION + 1)
    with pytest.raises(SolverBuiltError):
        val001_order.solve(12, 6)


@pytest.mark.unit
def test_saved_solve_is_not_reused_under_another_pressure_solver_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A file saved under one PRESSURE_SOLVER_VERSION builds a solver under another.

    Review 37 S4: the key caught ECR-003 only because pressure_rtol replaced
    pressure_tol; a later solve that keeps the keys would not have changed it.
    Defect caught: the version dropped from the key.
    """

    class SolverBuiltError(Exception):
        pass

    def no_solver(*args: object) -> None:
        raise SolverBuiltError

    monkeypatch.setattr(val001_order, "FIELD_DIR", tmp_path)
    monkeypatch.setattr(val001_order, "StaggeredSolver", no_solver)
    config = val001_order.load_case("poiseuille", grid=(12, 6))
    params = val001_order.reuse_key(config)
    np.savez(tmp_path / "poiseuille_12x6.npz", params=params, u=np.ones((6, 12)))
    assert np.array_equal(val001_order.solve(12, 6)["u"], np.ones((6, 12)))
    monkeypatch.setattr(
        val001_order,
        "PRESSURE_SOLVER_VERSION",
        val001_order.PRESSURE_SOLVER_VERSION + 1,
    )
    with pytest.raises(SolverBuiltError):
        val001_order.solve(12, 6)
