"""Unit tests for src/stopping.py on synthetic step histories.

Each test names the planted defect it was shown to catch.
"""

import math

import numpy as np
import pytest

from src.stopping import RATE_WINDOW, ErrorEstimateRule

SCALE = 0.1


def _feed(
    rule: ErrorEstimateRule, steps: np.ndarray, imbalance: float = 0.0
) -> list[bool]:
    """Feed a history one step at a time with a fixed imbalance; each answer."""
    return [rule.update(float(s), lambda: imbalance) for s in steps]


def _geometric(rho: float, n: int, last: float) -> np.ndarray:
    """n steps falling by rho per iteration and ending at last."""
    return last * rho ** (np.arange(n) - (n - 1.0))


@pytest.mark.unit
@pytest.mark.parametrize("rho", [0.5, 0.9, 0.9999])
def test_estimate_is_step_rho_over_one_minus_rho(rho: float) -> None:
    """Defects caught: the step alone as the estimate; 1 + rho in the denominator."""
    rule = ErrorEstimateRule(SCALE, 1e-6, 1e-10)
    _feed(rule, _geometric(rho, RATE_WINDOW + 20, 1e-7))
    expected = 1e-7 * rho / (1.0 - rho) / SCALE
    assert rule.estimate_history[-1] == pytest.approx(expected, rel=1e-9)


@pytest.mark.unit
def test_small_steps_at_a_slow_rate_do_not_converge() -> None:
    """False convergence: steps of 1e-9 of the scale at rho = 0.9999 leave 1e-5.

    The same history passes a step test at 1e-6 from its first entry.
    """
    steps = _geometric(0.9999, 3 * RATE_WINDOW, 1e-9 * SCALE)
    assert np.all(steps / SCALE < 1e-6)
    rule = ErrorEstimateRule(SCALE, 1e-6, 1e-10)
    assert not any(_feed(rule, steps))
    assert rule.estimate_history[-1] == pytest.approx(1e-9 * 0.9999 / 1e-4)


@pytest.mark.unit
def test_no_estimate_one_short_of_the_window_and_one_at_exactly_the_window() -> None:
    """Defects caught: the window guard's < as <=, and as < RATE_WINDOW - 1."""
    rule = ErrorEstimateRule(SCALE, 1e-6, 1e-10)
    answers = _feed(rule, _geometric(0.5, RATE_WINDOW, 1e-30))
    assert answers == [False] * (RATE_WINDOW - 1) + [True]


@pytest.mark.unit
@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("kind", ["flat", "growing", "zero", "nan"])
def test_a_stall_never_converges(kind: str) -> None:
    """Defects caught: the rho_hat range check removed; the step guard removed.

    A zero or NaN sits in the only full window. Without the step guard the
    range check still refuses it, but only after log(0) warns, which this
    test makes an error. A flat history's slope is zero only to rounding, so
    its estimate may be huge rather than inf.
    """
    n = RATE_WINDOW + 10
    steps = {
        "flat": np.full(n, 1e-12),
        "growing": _geometric(1.01, n, 1e-12),
    }.get(kind, _geometric(0.5, RATE_WINDOW, 1e-30))
    if kind in ("zero", "nan"):
        steps[-40] = 0.0 if kind == "zero" else np.nan
    rule = ErrorEstimateRule(SCALE, 1e-6, 1e-10)
    assert not any(_feed(rule, steps))
    assert kind == "flat" or rule.estimate_history[-1] == math.inf


@pytest.mark.unit
@pytest.mark.parametrize(("imbalance", "expected"), [(2e-10, False), (5e-11, True)])
def test_continuity_decides_once_the_estimate_is_met(
    imbalance: float, expected: bool
) -> None:
    """Defect caught: condition (b) dropped. The imbalance is asked for only under (a)."""
    calls: list[float] = []

    def worst() -> float:
        calls.append(imbalance)
        return imbalance

    rule = ErrorEstimateRule(SCALE, 1e-6, 1e-10)
    steps = _geometric(0.5, RATE_WINDOW + 5, 1e-30)
    assert [rule.update(float(s), worst) for s in steps][-1] is expected
    assert len(calls) == 6


@pytest.mark.unit
@pytest.mark.parametrize("position", [0, 1, 2])
@pytest.mark.parametrize("bad", [0.0, -1.0, math.inf, math.nan, True])
def test_a_scale_or_tolerance_not_positive_and_finite_is_rejected(
    position: int, bad: float
) -> None:
    """A zero scale would divide by zero, a NaN tolerance is never met, True is not 1.0."""
    args = [SCALE, 1e-6, 1e-10]
    args[position] = bad
    with pytest.raises(ValueError, match="must be positive and finite"):
        ErrorEstimateRule(*args)
