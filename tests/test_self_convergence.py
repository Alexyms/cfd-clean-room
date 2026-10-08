"""Unit tests for the self-convergence instrument in scripts/self_convergence.py.

The control is the instrument's credibility, so it is tested both ways: it
recovers the orders it was built with, and it refuses to pass when the fields
carry an order other than the one it is told to expect.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import self_convergence  # noqa: E402 -- scripts/ is not a package; path set above

from src.config import SimConfig  # noqa: E402 -- follows sys.path.insert
from src.mesh import Mesh  # noqa: E402 -- follows sys.path.insert
from src.pressure import (  # noqa: E402 -- follows sys.path.insert
    PRESSURE_SOLVER_VERSION,
    STAGGERED_METHODS,
)
from src.stopping import IterationState  # noqa: E402 -- follows sys.path.insert
from validation.cases import load_case  # noqa: E402 -- follows sys.path.insert
from validation.metrics import (  # noqa: E402 -- follows sys.path.insert
    GHIA_U_VAL,
    GHIA_U_Y,
    GHIA_V_VAL,
    GHIA_V_X,
    MARCHI_U_ROWS,
    MARCHI_V_ROWS,
    cavity_centerline_profiles,
    cavity_true_centerline_profiles,
)


@pytest.mark.unit
def test_control_recovers_orders_two_one_and_one_half() -> None:
    """Every estimate is within 0.05 of the q the synthetic fields carry."""
    results = self_convergence.run_control()
    assert set(results) == {"2.0", "1.0", "0.5"}
    for q, estimates in results.items():
        assert len(estimates) == 14
        assert all(
            abs(p - float(q)) < self_convergence.CONTROL_TOL for p in estimates.values()
        )


@pytest.mark.unit
def test_control_fails_on_an_order_it_was_not_told(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The planted failure: fields half an order off what the control expects must stop it."""
    real = self_convergence.synthetic
    monkeypatch.setattr(
        self_convergence, "synthetic", lambda n, q, c: real(n, q - 0.5, c)
    )
    with pytest.raises(SystemExit, match="control failed"):
        self_convergence.run_control()


@pytest.mark.unit
def test_restriction_averages_each_two_by_two_block() -> None:
    """R maps a 4x4 field to the 2x2 block means, exactly."""
    fine = np.arange(16.0).reshape(4, 4)
    assert np.array_equal(
        self_convergence.restrict(fine), np.array([[2.5, 4.5], [10.5, 12.5]])
    )


@pytest.mark.unit
def test_centerline_faces_recover_the_faces_exactly() -> None:
    """Cell means of known staggered faces give those faces back, and a zero far wall."""
    rng = np.random.default_rng(0)
    n = 8
    uf = rng.standard_normal((n, n + 1))
    vf = rng.standard_normal((n + 1, n))
    uf[:, 0] = uf[:, n] = 0.0
    vf[0, :] = vf[n, :] = 0.0
    u = 0.5 * (uf[:, :-1] + uf[:, 1:])
    v = 0.5 * (vf[:-1, :] + vf[1:, :])
    u_line, v_line, residual = self_convergence.centerline_faces(u, v)
    assert np.allclose(u_line, uf[:, n // 2], rtol=0.0, atol=1e-13)
    assert np.allclose(v_line, vf[n // 2, :], rtol=0.0, atol=1e-13)
    assert residual < 1e-13


def _gap_orders(
    fields: dict[int, tuple[np.ndarray, np.ndarray]],
) -> dict[str, np.ndarray]:
    """Observed orders of the largest profile-to-face gap, per sampling, [u, v] per step."""
    gaps: dict[str, list[list[float]]] = {"true": [], "offset": []}
    for n in self_convergence.GRIDS:
        config = load_case("cavity", grid=(n, n))
        mesh = Mesh(config)
        for name, profiles in (
            ("true", cavity_true_centerline_profiles),
            ("offset", cavity_centerline_profiles),
        ):
            g = self_convergence.face_gaps(config, mesh, *fields[n], profiles)
            gaps[name].append([g["u"], g["v"]])
    return {k: np.log2(np.divide(g[:-1], g[1:])) for k, g in gaps.items()}


@pytest.mark.unit
def test_true_centerline_meets_smooth_faces_at_second_order() -> None:
    """Known smooth faces: the true-centerline gap is order 2, the offset one order 1.

    f vanishes on both walls, as the recovery in centerline_faces needs, and has
    a nonzero slope and curvature at 0.5, so neither order is degenerate. The
    faces do not vary along the centerline: a factor that did would put each
    grid's largest gap at a different row and shift every order by about 0.03.
    """

    def f(s: np.ndarray) -> np.ndarray:
        t = s - 0.5
        return 0.1 * np.sin(np.pi * s) + t * (1.0 - 4.0 * t**2)

    fields = {}
    for n in self_convergence.GRIDS:
        uf = np.tile(f(np.arange(n + 1) / n), (n, 1))
        u = 0.5 * (uf[:, :-1] + uf[:, 1:])
        fields[n] = (u, u.T.copy())
    orders = _gap_orders(fields)
    tol = self_convergence.CONTROL_TOL
    assert np.all(np.abs(orders["true"] - 2.0) < tol), orders
    assert np.all(np.abs(orders["offset"] - 1.0) < tol), orders


def _smooth_faces(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Cell-centered u, v of the smooth faces used above, on an n x n grid."""

    def f(s: np.ndarray) -> np.ndarray:
        t = s - 0.5
        return 0.1 * np.sin(np.pi * s) + t * (1.0 - 4.0 * t**2)

    uf = np.tile(f(np.arange(n + 1) / n), (n, 1))
    u = 0.5 * (uf[:, :-1] + uf[:, 1:])
    return u, u.T.copy()


def _save_fields(
    directory: Path, fields: dict[int, tuple[np.ndarray, np.ndarray]]
) -> None:
    """Write fields in the layout solve_and_save and solve_tight leave under results/.

    The tight file holds the same field at every snapshot tolerance, so the
    station tables are computed on it and reveal nothing about the solver.
    """
    case_tol = load_case("cavity").convergence_tol
    tags = [
        f"{t:.0e}"
        for t in (case_tol, *self_convergence.SNAPSHOT_TOLS, self_convergence.TIGHT_TOL)
    ]
    for n, (u, v) in fields.items():
        np.savez(directory / f"{self_convergence.STAGGERED_METHOD}_{n}.npz", u=u, v=v)
        snapshots = {
            key: value
            for t in tags
            for key, value in ((f"u_{t}", u), (f"v_{t}", v), (f"outer_{t}", 7))
        }
        np.savez(
            directory / f"{self_convergence.STAGGERED_METHOD}_{n}_tol1e-9.npz",
            seconds=1.5,
            **snapshots,
        )


def _saved_gap_orders(directory: Path, monkeypatch: pytest.MonkeyPatch) -> np.ndarray:
    """Order of the true-centerline u gap per refinement step, read back from extrapolation."""
    monkeypatch.setattr(self_convergence, "FIELD_DIR", directory)
    out = self_convergence.extrapolation()
    gaps = [
        out["metric"][f"{self_convergence.STAGGERED_METHOD}_{n}"]["face_gap"]["u"]
        for n in self_convergence.GRIDS
    ]
    return np.log2(np.divide(gaps[:-1], gaps[1:]))


@pytest.mark.integration
def test_saved_staggered_faces_are_met_at_second_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Saved fields of a known order come back from extrapolation at that order.

    Issue 47 D1: the test that read the real saved fields skipped whenever
    results/self_convergence was absent, so it had never run in CI. The fields
    here are the smooth faces of the test above, written in the layout the
    script saves, whose true-centerline gap is second order.
    """
    _save_fields(tmp_path, {n: _smooth_faces(n) for n in self_convergence.GRIDS})
    orders = _saved_gap_orders(tmp_path, monkeypatch)
    assert np.all(np.abs(orders - 2.0) < self_convergence.CONTROL_TOL), orders


@pytest.mark.integration
def test_saved_fields_of_another_order_fail_the_gap_order_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The planted wrong order: fields with a first-order error move the order off 2.

    A perturbation of size 1/n in every cell is first order in the gap. If the
    check above could not fail, it would certify the instrument regardless of
    the fields it reads.
    """
    fields = {}
    for n in self_convergence.GRIDS:
        u, _v = _smooth_faces(n)
        bump = np.cos(np.pi * (np.arange(n) + 0.5) / n)[None, :] / n
        fields[n] = (u + bump, (u + bump).T.copy())
    _save_fields(tmp_path, fields)
    orders = _saved_gap_orders(tmp_path, monkeypatch)
    assert not np.all(np.abs(orders - 2.0) < self_convergence.CONTROL_TOL), orders
    assert np.all(np.abs(orders - 1.0) < 0.2), orders


@pytest.mark.integration
def test_face_gap_matches_the_mean_of_the_two_middle_columns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """face_gap on the 20x20 grid equals the middle-column mean minus the exact face.

    Issue 47 D2: the entry was written for every saved field and asserted
    nowhere. On the smooth faces u varies along x only and the exact face on
    x = 0.5 is f(0.5) = 0.1. The true-centerline profile is the mean of the
    two columns whose centers bracket 0.5 (columns 9 and 10 of 20), read here
    from the file that was saved, so the gap is that mean minus 0.1 in every
    row, and the same for v by symmetry.
    """
    fields = {n: _smooth_faces(n) for n in self_convergence.GRIDS}
    _save_fields(tmp_path, fields)
    monkeypatch.setattr(self_convergence, "FIELD_DIR", tmp_path)
    out = self_convergence.extrapolation()
    entry = out["metric"][f"{self_convergence.STAGGERED_METHOD}_20"]["face_gap"]
    with np.load(tmp_path / f"{self_convergence.STAGGERED_METHOD}_20.npz") as saved:
        middle = 0.5 * (saved["u"][:, 9] + saved["u"][:, 10])
    expected = float(np.abs(middle - 0.1).max())
    assert expected > 1e-6
    assert entry["u"] == pytest.approx(expected, rel=1e-9)
    assert entry["v"] == pytest.approx(expected, rel=1e-9)


@pytest.mark.unit
def test_extrapolation_control_recovers_a_known_limit_and_refuses_first_order() -> None:
    """h^2 returns the limit to rounding at order 2; h reads order 1 and is unlicensed."""
    stations = np.array(GHIA_V_X[1:-1])
    self_convergence.run_extrapolation_control(stations)
    second = self_convergence.richardson(
        self_convergence.synthetic_profiles(2.0), stations
    )
    first = self_convergence.richardson(
        self_convergence.synthetic_profiles(1.0), stations
    )
    exact = stations**3 - 0.5 * stations
    assert np.allclose(second["order"], 2.0, rtol=0.0, atol=1e-9)
    assert np.allclose(second["limit"], exact, rtol=0.0, atol=1e-13)
    assert np.allclose(first["order"], 1.0, rtol=0.0, atol=1e-9)
    assert np.isnan(first["limit"]).all()


@pytest.mark.unit
def test_order_change_names_a_station_undefined_under_one_reading() -> None:
    """A NaN on one side is listed by station, not dropped from the maximum."""
    stations = np.array([0.1, 0.2, 0.3])
    change = self_convergence.order_change(
        np.array([2.0, np.nan, 2.5]), np.array([2.1, 2.0, 2.0]), stations
    )
    assert change["max"] == pytest.approx(0.5)
    assert change["at"] == 0.3
    assert change["undefined_in_one"] == [0.2]


@pytest.mark.unit
def test_extrapolation_control_stops_on_a_wrong_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The planted failure: a profile whose limit is not the one the control expects."""
    real = self_convergence.synthetic_profiles
    monkeypatch.setattr(
        self_convergence,
        "synthetic_profiles",
        lambda q: {n: (s, f + 1e-6) for n, (s, f) in real(q).items()},
    )
    with pytest.raises(SystemExit, match="extrapolation control failed"):
        self_convergence.run_extrapolation_control(np.array(GHIA_V_X[1:-1]))


def _cells(p: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cell means of faces u = sin(pi x) p(yc), v = sin(pi y) q(xc): p, q on the midlines."""
    faces = np.arange(len(p) + 1) / len(p)
    uf = np.sin(np.pi * faces)[None, :] * p[:, None]
    vf = np.sin(np.pi * faces)[:, None] * q[None, :]
    return 0.5 * (uf[:, :-1] + uf[:, 1:]), 0.5 * (vf[:-1, :] + vf[1:, :])


class _ScriptedSolver:
    """Stands in for StaggeredSolver: residual 5e-6 / 10^k and u = v = k at iteration k."""

    def __init__(self, mesh: Mesh, config: SimConfig, boundary: object) -> None:
        self.config = config

    def solve_steady(self, on_iteration: Callable[[IterationState], None]) -> None:
        n = self.config.nx
        for k in range(self.config.max_simple_iter):
            field, residual = np.full((n, n), float(k)), 5e-6 * 10.0**-k
            on_iteration(IterationState(k, residual, 0, field, field, field, 0))
            if residual < self.config.convergence_tol:
                return


@pytest.mark.unit
@pytest.mark.parametrize(
    ("cap", "ulps", "component", "stops"),
    [
        (99, 0, "u", None),
        (99, 1, "u", "differs from the saved field"),
        (99, 1, "v", "differs from the saved field"),
        (3, 0, "u", "8x8 stopped before reaching 1e-08"),
    ],
)
def test_solve_tight_keeps_only_a_whole_continuation_of_the_saved_field(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cap: int,
    ulps: int,
    component: str,
    stops: str | None,
) -> None:
    """One ulp off the saved u or v, or a cap before 1e-9, stops it; else all is kept.

    Test 21c F2: only u was perturbed, so dropping the v comparison from the
    bitwise control passed every test.
    """
    monkeypatch.setattr(self_convergence, "FIELD_DIR", tmp_path)
    monkeypatch.setattr(self_convergence, "StaggeredSolver", _ScriptedSolver)
    monkeypatch.setattr(self_convergence, "TIGHT_MAX_ITER", cap)
    fields = {
        "u": np.ones((8, 8)),
        "v": np.ones((8, 8)),
    }  # iteration 1 is the first below 1e-6
    fields[component][0, 0] = np.nextafter(1.0, 2.0) if ulps else 1.0
    np.savez(tmp_path / "staggered-cg_8.npz", **fields)
    if stops:
        with pytest.raises(SystemExit, match=stops):
            self_convergence.solve_tight(8)
        return
    with np.load(self_convergence.solve_tight(8)) as data:
        tags = ("1e-06", "1e-07", "1e-08", "1e-09")
        assert [int(data[f"outer_{t}"]) for t in tags] == [2, 3, 4, 5]
        assert np.array_equal(data["u_1e-09"], np.full((8, 8), 4.0))


@pytest.mark.unit
def test_solve_tight_and_solve_and_save_skip_a_field_another_solver_saved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With a staggered-jacobi field saved for the grid, both solve and file under staggered-cg.

    Test 37 T-B1 (b): no test put a Jacobi-era file in FIELD_DIR, so
    solve_tight spelling the old label could return it unsolved. The name is
    the current PRESSURE_SOLVER_VERSION's label, which is what keys these
    files. Defect caught: either function reading or writing another
    version's file name.
    """
    monkeypatch.setattr(self_convergence, "FIELD_DIR", tmp_path)
    monkeypatch.setattr(self_convergence, "StaggeredSolver", _ScriptedSolver)
    monkeypatch.setattr(self_convergence, "TIGHT_MAX_ITER", 99)
    label = STAGGERED_METHODS[PRESSURE_SOLVER_VERSION]
    decoy = np.full((8, 8), -7.0)
    for version, old in STAGGERED_METHODS.items():
        if version != PRESSURE_SOLVER_VERSION:
            np.savez(tmp_path / f"{old}_8.npz", u=decoy, v=decoy, p=decoy)
            tags = ("1e-06", "1e-07", "1e-08", "1e-09")
            np.savez(
                tmp_path / f"{old}_8_tol1e-9.npz",
                **{f"{c}_{t}": decoy for c in "uv" for t in tags},
            )
    np.savez(tmp_path / f"{label}_8.npz", u=np.ones((8, 8)), v=np.ones((8, 8)))
    tight = self_convergence.solve_tight(8)
    assert tight == tmp_path / f"{label}_8_tol1e-9.npz"
    with np.load(tight) as data:
        assert np.array_equal(data["u_1e-09"], np.full((8, 8), 4.0))
    assert np.array_equal(self_convergence.tight_field(8)[0], np.full((8, 8), 4.0))

    class BuiltError(Exception):
        pass

    def spy(mesh: Mesh, config: SimConfig, boundary: object) -> None:
        raise BuiltError

    (tmp_path / f"{label}_8.npz").unlink()
    monkeypatch.setattr(self_convergence, "StaggeredSolver", spy)
    with pytest.raises(BuiltError):
        self_convergence.solve_and_save(label, 8)


@pytest.mark.unit
def test_solve_tight_checks_for_the_saved_field_before_it_solves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no saved field it stops at once, without building the solver (review 21b S5).

    The continuation takes minutes to hours; the check used to come after it, so
    a missing field was reported only when the solve had finished and its
    result was about to be thrown away.
    """

    class BuiltError(Exception):
        pass

    def spy(mesh: Mesh, config: SimConfig, boundary: object) -> None:
        raise BuiltError

    monkeypatch.setattr(self_convergence, "FIELD_DIR", tmp_path)
    monkeypatch.setattr(self_convergence, "StaggeredSolver", spy)
    with pytest.raises(SystemExit, match=r"staggered-cg_8\.npz is missing"):
        self_convergence.solve_tight(8)


@pytest.mark.unit
def test_every_solve_is_pinned_to_velocity_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The saved fields were taken under velocity_step; the case file now names another."""
    assert load_case("cavity").stopping_rule == "error_estimate"
    rules: list[str] = []

    class BuiltError(Exception):
        pass

    def spy(mesh: Mesh, config: SimConfig, boundary: object) -> None:
        rules.append(config.stopping_rule)
        raise BuiltError

    monkeypatch.setattr(self_convergence, "FIELD_DIR", tmp_path)
    monkeypatch.setattr(self_convergence, "StaggeredSolver", spy)
    for method in self_convergence.METHODS:
        with pytest.raises(BuiltError):
            self_convergence.solve_and_save(method, 8)
    ones = np.ones((8, 8))
    np.savez(tmp_path / f"{self_convergence.STAGGERED_METHOD}_8.npz", u=ones, v=ones)
    with pytest.raises(BuiltError):
        self_convergence.solve_tight(8)
    # One solve per method, staggered-cg alone since the retirement, then
    # solve_tight.
    assert rules == ["velocity_step"] * 2


@pytest.mark.unit
def test_face_profiles_divide_by_the_lid_and_take_nodes_from_the_mesh() -> None:
    """At a lid speed of 2 every value halves and the lid reads 1; nodes are the mesh's."""
    c, pos = (np.arange(8) + 0.5) / 8, np.arange(8.0)
    u, v = _cells(2.0 * c, 2.0 * c * (1.0 - c))
    mesh = SimpleNamespace(x=[-0.5, 1.5], y=[-1.0, 2.0], xc=pos**2, yc=pos**3)
    (y, u_line), (x, v_line) = self_convergence.face_profiles(mesh, u, v, 2.0)
    assert np.array_equal(y, [-1.0, *pos**3, 2.0])
    assert np.array_equal(x, [-0.5, *pos**2, 1.5])
    assert np.allclose(u_line, [0.0, *c, 1.0], rtol=0.0, atol=1e-14)
    assert np.allclose(v_line, [0.0, *(c * (1.0 - c)), 0.0], rtol=0.0, atol=1e-14)


@pytest.mark.unit
def test_station_orders_keep_each_interpolation_under_its_own_key() -> None:
    """No polynomial reproduces this profile, so the quintic orders differ from the cubic."""
    profiles = {}
    for n in self_convergence.GRIDS:
        s = np.array([0.0, *(np.arange(n) + 0.5) / n, 1.0])
        profiles[n] = (s, np.sin(3.0 * s) + np.cos(2.0 * s) / n**2)
    out = self_convergence.station_orders(profiles, GHIA_U_Y, GHIA_U_VAL)
    stations = np.array(GHIA_U_Y[1:-1])
    cubic = self_convergence.richardson(profiles, stations)
    quintic = self_convergence.richardson(profiles, stations, k=6)
    assert not np.allclose(cubic["order"], quintic["order"], rtol=0.0, atol=1e-6)
    assert out["station"] == stations.tolist()
    np.testing.assert_array_equal(out["order"], cubic["order"])
    np.testing.assert_array_equal(out["order_quintic"], quintic["order"])
    assert out["licensed_quintic"] == quintic["licensed"].tolist()


@pytest.mark.unit
def test_station_orders_returns_every_key_each_under_its_own_interpolation() -> None:
    """The cubic and quintic readings differ in what is licensed, and no key crosses over.

    Test 21c F1: licensed_quintic was asserted on a profile licensed under both,
    so swapping it with licensed passed, and order2_quintic_minus_ghia was
    asserted nowhere. sin(12 s) has enough curvature that the cubic reading is
    licensed at 5 of the 15 stations and the quintic at all 15. Every key is
    compared with a value formed here from richardson.
    """
    profiles = {}
    for n in self_convergence.GRIDS:
        s = np.array([0.0, *(np.arange(n) + 0.5) / n, 1.0])
        profiles[n] = (s, np.sin(12.0 * s) + (1.0 / n) ** 2 * (1.0 + s))
    out = self_convergence.station_orders(profiles, GHIA_U_Y, GHIA_U_VAL)
    stations, ref = np.array(GHIA_U_Y[1:-1]), np.array(GHIA_U_VAL[1:-1])
    cubic = self_convergence.richardson(profiles, stations)
    quintic = self_convergence.richardson(profiles, stations, k=6)
    assert set(out) == {
        "station",
        "order",
        "order_quintic",
        "licensed",
        "licensed_quintic",
        "f80_minus_ghia",
        "step_40_to_80",
        "order2_minus_ghia",
        "order2_quintic_minus_ghia",
        "interpolation_estimate",
        "quintic_order_change",
    }
    assert out["licensed"] == cubic["licensed"].tolist()
    assert out["licensed_quintic"] == quintic["licensed"].tolist()
    assert sum(out["licensed"]) == 5 and sum(out["licensed_quintic"]) == 15
    assert out["licensed"] != out["licensed_quintic"]
    np.testing.assert_array_equal(out["order"], cubic["order"])
    np.testing.assert_array_equal(out["order_quintic"], quintic["order"])
    np.testing.assert_array_equal(out["f80_minus_ghia"], cubic["f80"] - ref)
    np.testing.assert_array_equal(out["step_40_to_80"], cubic["f80"] - cubic["f40"])
    np.testing.assert_array_equal(out["order2_minus_ghia"], cubic["value"] - ref)
    np.testing.assert_array_equal(
        out["order2_quintic_minus_ghia"], quintic["value"] - ref
    )
    assert not np.allclose(
        out["order2_minus_ghia"], out["order2_quintic_minus_ghia"], rtol=0.0, atol=1e-6
    )
    for n in self_convergence.GRIDS:
        shift = np.abs(quintic[f"f{n}"] - cubic[f"f{n}"])
        got = out["interpolation_estimate"][n]
        assert got == {"max": float(shift.max()), "at": float(stations[shift.argmax()])}
    assert out["quintic_order_change"] == self_convergence.order_change(
        cubic["order"], quintic["order"], stations
    )


@pytest.mark.unit
def test_extrapolation_carries_known_orders_through_to_its_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A different known order q at each tolerance, from the saved files to the dict.

    The midline faces are cubics, y^3 + h^q y(1 - y) and x(1 - x)(x + h^q), so every
    order is q exactly and the order-2 value at q = 2 is the limit.
    """
    monkeypatch.setattr(self_convergence, "FIELD_DIR", tmp_path)
    q = {"1e-06": 1.0, "1e-07": 1.5, "1e-08": 2.0, "1e-09": 2.1}
    fields, fields_v = {}, {}
    for n in self_convergence.GRIDS:
        snaps: dict[str, np.ndarray] = {}
        c = (np.arange(n) + 0.5) / n
        for t in q:
            e = float(n) ** -q[t]
            u, v = _cells(c**3 + e * c * (1 - c), c * (1 - c) * (c + e))
            fields[n, t], fields_v[n, t] = u, v
            snaps |= {f"u_{t}": u, f"v_{t}": v, f"outer_{t}": np.array(1)}
        np.savez(tmp_path / f"staggered-cg_{n}_tol1e-9.npz", seconds=1.0, **snaps)
        for method in self_convergence.METHODS:
            np.savez(
                tmp_path / f"{method}_{n}.npz",
                u=fields[n, "1e-06"],
                v=fields_v[n, "1e-06"],
            )
    out = self_convergence.extrapolation()
    for axis, positions, ghia, limit in (
        ("u", GHIA_U_Y, GHIA_U_VAL, lambda s: s**3),
        ("v", GHIA_V_X, GHIA_V_VAL, lambda s: s**2 * (1 - s)),
    ):
        stations = np.array(positions[1:-1])
        for tag in q:
            got = out["stations"][tag][axis]
            assert got["station"] == stations.tolist()
            assert np.allclose(got["order"], q[tag], rtol=0.0, atol=1e-8)
            assert np.allclose(got["order_quintic"], q[tag], rtol=0.0, atol=1e-8)
        expected = limit(stations) - np.array(ghia[1:-1])
        got = out["stations"]["1e-08"][axis]["order2_minus_ghia"]
        assert np.allclose(got, expected, rtol=0.0, atol=1e-12)
        assert out["settling"][axis]["max"] == pytest.approx(0.1, abs=1e-8)
    for n in self_convergence.GRIDS:
        change = np.abs(fields[n, "1e-09"] - fields[n, "1e-06"]).max()
        assert out["iteration"][n]["u_change"] == change
        change_v = np.abs(fields_v[n, "1e-09"] - fields_v[n, "1e-06"]).max()
        assert out["iteration"][n]["v_change"] == change_v
        assert change_v != change


# The Marchi stations paired with a Ghia station under GHIA_PAIR = 3/128, as the
# tables in docs/reports/cavity_reference_marchi.md list them.
MARCHI_GHIA_PAIRS = {
    "u": {
        0.0625: 0.0625,
        0.125: 0.1016,
        0.1875: 0.1719,
        0.4375: 0.4531,
        0.5: 0.5,
        0.625: 0.6172,
        0.75: 0.7344,
        0.875: 0.8516,
        0.9375: 0.9531,
    },
    "v": {
        0.0625: 0.0625,
        0.25: 0.2344,
        0.5: 0.5,
        0.8125: 0.8047,
        0.875: 0.8594,
        0.9375: 0.9453,
    },
}
MARCHI_CASES = (
    ("u", MARCHI_U_ROWS, GHIA_U_Y, GHIA_U_VAL, lambda s: s**3),
    ("v", MARCHI_V_ROWS, GHIA_V_X, GHIA_V_VAL, lambda s: s**2 * (1 - s)),
)


def _marchi_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Fields of exact order 2 at 20, 40, 80 and 100, saved where tight_field looks.

    The midline faces are y^3 + h^2 y(1 - y) and x(1 - x)(x + h^2). Both are
    cubics, which both interpolations reproduce, so R(40, 80) and R(80, 100)
    return the limits y^3 and x^2(1 - x) to rounding. Every grid goes in
    FIELD_DIR under the current label; a Jacobi-era decoy sits beside them
    under the retired label, and a 1e-08 snapshot of order 1 in every file.
    Returns the order-2 fields by grid.
    """
    builder = tmp_path / "builder"
    builder.mkdir()
    monkeypatch.setattr(self_convergence, "FIELD_DIR", builder)

    def cells(n: int, q: float) -> tuple[np.ndarray, np.ndarray]:
        c, e = (np.arange(n) + 0.5) / n, float(n) ** -q
        return _cells(c**3 + e * c * (1 - c), c * (1 - c) * (c + e))

    fields = {}
    for n in (*self_convergence.GRIDS, 100):
        fields[n] = cells(n, 2.0)
        decoy = cells(n, 1.0)
        snaps = {"u_1e-09": fields[n][0], "v_1e-09": fields[n][1]}
        snaps |= {"u_1e-08": decoy[0], "v_1e-08": decoy[1]}
        np.savez(builder / f"staggered-cg_{n}_tol1e-9.npz", **snaps)
    u, v = cells(80, 1.0)
    np.savez(
        builder / "staggered-jacobi_80_tol1e-9.npz", **{"u_1e-09": u, "v_1e-09": v}
    )
    return fields


@pytest.mark.unit
def test_tight_field_reads_the_current_labels_1e_9_snapshot_and_nothing_else(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """80 and 100 come from the staggered-cg files; the Jacobi-era decoy and the
    1e-08 snapshots are never read, and a grid with no file raises rather
    than falling back to a field of unknown solver."""
    fields = _marchi_fields(tmp_path, monkeypatch)
    for n in (80, 100):
        u, v = self_convergence.tight_field(n)
        assert np.array_equal(u, fields[n][0])
        assert np.array_equal(v, fields[n][1])
    with pytest.raises(FileNotFoundError):
        self_convergence.tight_field(60)
    assert not hasattr(self_convergence, "TESTER_DIR")
    assert self_convergence.METHODS == ("staggered-cg",)


@pytest.mark.unit
def test_marchi_comparison_returns_the_known_limit_from_both_pairs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Order 2 everywhere; R(40, 80), R(80, 100) and its quintic land on the limit."""
    _marchi_fields(tmp_path, monkeypatch)
    out = self_convergence.marchi_comparison()
    for axis, rows, _positions, _values, limit in MARCHI_CASES:
        s, ref, err = (np.array(c) for c in zip(*rows, strict=True))
        got = out[axis]
        assert (got["station"], got["marchi"]) == (s.tolist(), ref.tolist())
        assert got["marchi_error"] == err.tolist()
        assert np.allclose(got["order"], 2.0, rtol=0.0, atol=1e-8)
        for key in (
            "r40_80_minus_marchi",
            "r80_100_minus_marchi",
            "r80_100_quintic_minus_marchi",
        ):
            assert np.allclose(got[key], limit(s) - ref, rtol=0.0, atol=1e-12), key
        f100 = limit(s) + 1e-4 * s * (1 - s)
        assert np.allclose(got["f100_minus_marchi"], f100 - ref, rtol=0.0, atol=1e-12)
        assert max(got["interpolation_estimate"].values()) < 1e-12


@pytest.mark.unit
def test_marchi_comparison_pairs_ghia_and_carries_it_along_the_extrapolation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The report's pairs, each Ghia value carried by limit(station) - limit(Ghia's).

    The limits differ between paired stations, so a wrong carry moves every
    pair except the shared stations 0.0625 and 0.5, where the carry is zero.
    """
    _marchi_fields(tmp_path, monkeypatch)
    out = self_convergence.marchi_comparison()
    for axis, rows, positions, values, limit in MARCHI_CASES:
        ghia = dict(zip(positions, values, strict=True))
        pairs, got = MARCHI_GHIA_PAIRS[axis], out[axis]
        for k, s in enumerate(got["station"]):
            entry = (
                got["ghia_station"][k],
                got["carry"][k],
                got["ghia_minus_marchi"][k],
            )
            if s not in pairs:
                assert entry == (None, None, None), (axis, s)
                continue
            g, carry = pairs[s], limit(s) - limit(pairs[s])
            assert entry[0] == g
            assert entry[1] == pytest.approx(carry, rel=0.0, abs=1e-12)
            assert entry[2] == pytest.approx(
                ghia[g] + carry - rows[k][1], rel=0.0, abs=1e-12
            )
