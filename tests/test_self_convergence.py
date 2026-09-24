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
from src.solver_ns import IterationState  # noqa: E402 -- follows sys.path.insert
from validation.cases import load_case  # noqa: E402 -- follows sys.path.insert
from validation.metrics import (  # noqa: E402 -- follows sys.path.insert
    GHIA_U_VAL,
    GHIA_U_Y,
    GHIA_V_VAL,
    GHIA_V_X,
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


@pytest.mark.integration
def test_true_centerline_meets_the_saved_staggered_faces_at_second_order() -> None:
    """The saved staggered solves: the gap is second order, the offset one first order.

    Each order is assigned to whichever of 1 and 2 it is nearer, which is the
    question asked, not a tolerance. The gaps are printed for the report. The
    fields are gitignored, so this skips where they have not been saved.
    """
    paths = {
        n: self_convergence.FIELD_DIR / f"staggered-jacobi_{n}.npz"
        for n in self_convergence.GRIDS
    }
    if not all(p.exists() for p in paths.values()):
        pytest.skip(
            "saved staggered fields not present; run scripts/self_convergence.py"
        )
    fields = {}
    for n, path in paths.items():
        with np.load(path) as data:
            fields[n] = (data["u"], data["v"])
    orders = _gap_orders(fields)
    print(f"gap orders [u, v] per step: {orders}")
    assert np.all(np.abs(orders["true"] - 2.0) < 0.5), orders
    assert np.all(np.abs(orders["offset"] - 1.0) < 0.5), orders


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
            on_iteration(IterationState(k, residual, 0, field, field, field))
            if residual < self.config.convergence_tol:
                return


@pytest.mark.unit
@pytest.mark.parametrize(
    ("cap", "ulps", "stops"),
    [
        (99, 0, None),
        (99, 1, "differs from the saved field"),
        (3, 0, "8x8 stopped before reaching 1e-08"),
    ],
)
def test_solve_tight_keeps_only_a_whole_continuation_of_the_saved_field(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cap: int, ulps: int, stops: str
) -> None:
    """One ulp off the saved field, or a cap before 1e-9, stops it; else all is kept."""
    monkeypatch.setattr(self_convergence, "FIELD_DIR", tmp_path)
    monkeypatch.setattr(self_convergence, "StaggeredSolver", _ScriptedSolver)
    monkeypatch.setattr(self_convergence, "TIGHT_MAX_ITER", cap)
    saved = np.ones((8, 8))  # iteration 1 is the first below 1e-6
    saved[0, 0] = np.nextafter(1.0, 2.0) if ulps else 1.0
    np.savez(tmp_path / "staggered-jacobi_8.npz", u=saved, v=np.ones((8, 8)))
    if stops:
        with pytest.raises(SystemExit, match=stops):
            self_convergence.solve_tight(8)
        return
    with np.load(self_convergence.solve_tight(8)) as data:
        tags = ("1e-06", "1e-07", "1e-08", "1e-09")
        assert [int(data[f"outer_{t}"]) for t in tags] == [2, 3, 4, 5]
        assert np.array_equal(data["u_1e-09"], np.full((8, 8), 4.0))


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
def test_extrapolation_carries_known_orders_through_to_its_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A different known order q at each tolerance, from the saved files to the dict.

    The midline faces are cubics, y^3 + h^q y(1 - y) and x(1 - x)(x + h^q), so every
    order is q exactly and the order-2 value at q = 2 is the limit.
    """
    monkeypatch.setattr(self_convergence, "FIELD_DIR", tmp_path)
    q = {"1e-06": 1.0, "1e-07": 1.5, "1e-08": 2.0, "1e-09": 2.1}
    fields = {}
    for n in self_convergence.GRIDS:
        snaps: dict[str, np.ndarray] = {}
        c = (np.arange(n) + 0.5) / n
        for t in q:
            e = float(n) ** -q[t]
            u, v = _cells(c**3 + e * c * (1 - c), c * (1 - c) * (c + e))
            fields[n, t] = u
            snaps |= {f"u_{t}": u, f"v_{t}": v, f"outer_{t}": np.array(1)}
        np.savez(tmp_path / f"staggered-jacobi_{n}_tol1e-9.npz", seconds=1.0, **snaps)
        for method in self_convergence.METHODS:
            np.savez(tmp_path / f"{method}_{n}.npz", u=fields[n, "1e-06"], v=v)
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
