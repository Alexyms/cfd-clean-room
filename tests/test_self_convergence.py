"""Unit tests for the self-convergence instrument in scripts/self_convergence.py.

The control is the instrument's credibility, so it is tested both ways: it
recovers the orders it was built with, and it refuses to pass when the fields
carry an order other than the one it is told to expect.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import self_convergence  # noqa: E402 -- scripts/ is not a package; path set above

from src.mesh import Mesh  # noqa: E402 -- follows sys.path.insert
from validation.cases import load_case  # noqa: E402 -- follows sys.path.insert
from validation.metrics import (  # noqa: E402 -- follows sys.path.insert
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


@pytest.mark.unit
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
