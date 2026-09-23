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
