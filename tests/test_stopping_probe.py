"""Unit tests for the estimator and the true-error pipeline of scripts/stopping_probe.py.

Each check is shown to catch the defect it exists for: the rate check rejects a
trailing window off by one iteration and a fit to the raw residual instead of
its log, and the true-error check tells a FLUID cell from a non-FLUID one.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import stopping_probe  # noqa: E402 -- scripts/ is not a package; path set above

from src.mesh import FLUID, Mesh  # noqa: E402 -- follows sys.path.insert
from validation.cases import load_case  # noqa: E402 -- follows sys.path.insert

WINDOW = 100


def rate_check(estimator: Callable[[np.ndarray, int], float]) -> bool:
    """True when an estimator recovers rho from two synthetic residual histories.

    A geometric history C rho^n has one rate, which any fit to its log returns.
    A history whose log is a n + b n^2 has a rate that drifts, and a straight-line
    fit to its log over indices m - (w - 1) / 2 to m + (w - 1) / 2 has slope
    exactly a + 2 b m, so a window shifted or resized by one index moves the
    answer by about b.
    """
    for rho in (0.5, 0.99, 0.999):
        history = 1e-3 * rho ** np.arange(400.0)
        if abs(estimator(history, WINDOW) - rho) > 1e-12:
            return False
    a, b, n = -0.01, 1e-5, np.arange(300.0)
    centre = n[-1] - (WINDOW - 1) / 2.0
    expected = np.exp(a + 2.0 * b * centre)
    return abs(estimator(np.exp(a * n + b * n**2), WINDOW) - expected) < 1e-12


def _fit(values: np.ndarray) -> float:
    return float(np.exp(np.polyfit(np.arange(len(values)), values, 1)[0]))


PLANTED = {
    "one entry too many": lambda h, w: _fit(np.log(h[-w - 1 :])),
    "one entry too few": lambda h, w: _fit(np.log(h[-w + 1 :])),
    "shifted back by one": lambda h, w: _fit(np.log(h[-w - 1 : -1])),
    "fit to raw values": lambda h, w: _fit(h[-w:]),
}


@pytest.mark.unit
def test_rho_hat_recovers_known_rates() -> None:
    """rho_hat returns the rate of a geometric history and of a drifting one."""
    assert rate_check(stopping_probe.rho_hat)


@pytest.mark.unit
@pytest.mark.parametrize("defect", sorted(PLANTED))
def test_rate_check_rejects_a_planted_defect(defect: str) -> None:
    """Each planted estimator defect fails the check rho_hat passes."""
    assert not rate_check(PLANTED[defect])


@pytest.mark.unit
def test_rho_hat_is_nan_on_a_history_shorter_than_the_window() -> None:
    """Too short a history gives no rate rather than a rate from fewer points."""
    assert np.isnan(stopping_probe.rho_hat(np.ones(WINDOW - 1), WINDOW))


@pytest.mark.unit
def test_estimate_is_the_geometric_tail() -> None:
    """step rho / (1 - rho) is the sum of every later step of a geometric iteration."""
    step, rho = 2.0**-20, 0.75
    tail = sum(step * rho**k for k in range(1, 200))
    assert stopping_probe.estimate(step, rho) == pytest.approx(tail, rel=1e-14)
    assert stopping_probe.estimate(step, 1.0) == float("inf")
    assert np.isnan(stopping_probe.estimate(step, float("nan")))


@pytest.fixture
def cavity() -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """The 20x20 cavity's FLUID mask and a dyadic truth, so differences are exact."""
    mesh = Mesh(load_case("cavity", grid=(20, 20)))
    rng = np.random.default_rng(23)
    truth = rng.integers(-1024, 1024, size=(2, 20, 20)) / 1024.0
    return mesh.cell_type == FLUID, (truth[0], truth[1])


@pytest.mark.unit
@pytest.mark.parametrize("component", [0, 1])
def test_true_error_returns_a_planted_offset_in_a_fluid_cell(
    cavity: tuple[np.ndarray, tuple[np.ndarray, np.ndarray]], component: int
) -> None:
    """Truth plus 2^-20 in one FLUID cell of u or v reads as exactly 2^-20."""
    fluid, truth = cavity
    assert fluid[5, 7]
    planted = [truth[0].copy(), truth[1].copy()]
    planted[component][5, 7] += 2.0**-20
    assert stopping_probe.true_error(*planted, truth, fluid) == 2.0**-20


@pytest.mark.unit
@pytest.mark.parametrize("component", [0, 1])
def test_true_error_ignores_an_offset_outside_the_fluid(
    cavity: tuple[np.ndarray, tuple[np.ndarray, np.ndarray]], component: int
) -> None:
    """The same offset in the BOUNDARY ring reads as zero."""
    fluid, truth = cavity
    assert not fluid[0, 7]
    planted = [truth[0].copy(), truth[1].copy()]
    planted[component][0, 7] += 2.0**-20
    assert stopping_probe.true_error(*planted, truth, fluid) == 0.0
