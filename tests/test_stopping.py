"""Unit tests for src/stopping.py on synthetic step histories.

Each test names the planted defect it was shown to catch.
"""

import math

import numpy as np
import pytest

from src.stopping import RATE_WINDOW, ErrorEstimateRule, ImbalanceSummary

SCALE = 0.1
# VAL-001's flux scale, rho 1 times 0.1 m/s through 0.5 m. Not 1, so a rule
# that forgot to divide by it would be caught.
FLUX = 0.05


def _rule() -> ErrorEstimateRule:
    """The rule at the default tolerances, 1e-6 and 1e-10."""
    return ErrorEstimateRule(SCALE, FLUX, 1e-6, 1e-10)


def _summary(
    worst: float = 0.0, absolute_sum: float = 0.0, signed_sum: float = 0.0
) -> ImbalanceSummary:
    """Imbalance readings, zero unless given."""
    return ImbalanceSummary(
        worst=worst, absolute_sum=absolute_sum, signed_sum=signed_sum
    )


def _feed(rule: ErrorEstimateRule, steps: np.ndarray, worst: float = 0.0) -> list[bool]:
    """Feed a history one step at a time with a fixed worst and no summed imbalance."""
    return [rule.update(float(s), lambda: _summary(worst)) for s in steps]


def _geometric(rho: float, n: int, last: float) -> np.ndarray:
    """n steps falling by rho per iteration and ending at last."""
    return last * rho ** (np.arange(n) - (n - 1.0))


@pytest.mark.unit
@pytest.mark.parametrize("rho", [0.5, 0.9, 0.9999])
def test_estimate_is_step_rho_over_one_minus_rho(rho: float) -> None:
    """Defects caught: the step alone as the estimate; 1 + rho in the denominator."""
    rule = _rule()
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
    rule = _rule()
    assert not any(_feed(rule, steps))
    assert rule.estimate_history[-1] == pytest.approx(1e-9 * 0.9999 / 1e-4)


@pytest.mark.unit
def test_no_estimate_one_short_of_the_window_and_one_at_exactly_the_window() -> None:
    """Defects caught: the window guard's < as <=, and as < RATE_WINDOW - 1."""
    rule = _rule()
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
    rule = _rule()
    assert not any(_feed(rule, steps))
    assert kind == "flat" or rule.estimate_history[-1] == math.inf


@pytest.mark.unit
@pytest.mark.parametrize(("imbalance", "expected"), [(2e-10, False), (5e-11, True)])
def test_continuity_decides_once_the_estimate_is_met(
    imbalance: float, expected: bool
) -> None:
    """Defect caught: condition (b) dropped. The imbalance is asked for only under (a)."""
    calls: list[float] = []

    def worst() -> ImbalanceSummary:
        calls.append(imbalance)
        return _summary(imbalance)

    rule = _rule()
    steps = _geometric(0.5, RATE_WINDOW + 5, 1e-30)
    assert [rule.update(float(s), worst) for s in steps][-1] is expected
    assert len(calls) == 6


@pytest.mark.unit
@pytest.mark.parametrize(("total", "expected"), [(1e-7, False), (2.5e-8, True)])
def test_summed_imbalance_decides_once_the_estimate_and_worst_cell_are_met(
    total: float, expected: bool
) -> None:
    """Defect caught: condition (c) dropped. Every cell is under 1e-10 in both.

    Over FLUX the sums are 2e-6 and 5e-7 against the tolerance 1e-6. A rule
    that did not divide by FLUX would pass both.
    """
    rule = _rule()
    steps = _geometric(0.5, RATE_WINDOW, 1e-30)
    assert [rule.update(float(s), lambda: _summary(5e-11, total)) for s in steps][
        -1
    ] is expected


@pytest.mark.unit
@pytest.mark.parametrize(
    ("signed", "expected"), [(2e-10, False), (5e-11, True), (-2e-10, False)]
)
def test_signed_domain_sum_decides_once_the_other_three_are_met(
    signed: float, expected: bool
) -> None:
    """Defects caught: condition (d) dropped; its absolute value dropped.

    Every cell is under 1e-10 and the absolute sum is 5e-7 of FLUX, so (a),
    (b) and (c) hold in all three. A net outflow and a net inflow of 2e-10
    fail criterion 6's domain-sum bound alike.
    """
    rule = _rule()
    steps = _geometric(0.5, RATE_WINDOW, 1e-30)
    summary = _summary(5e-11, 2.5e-8, signed)
    assert [rule.update(float(s), lambda: summary) for s in steps][-1] is expected


@pytest.mark.unit
def test_imbalance_summary_refuses_positional_readings() -> None:
    """Defect caught: kw_only removed, which lets two readings swap silently (test 28 S2)."""
    with pytest.raises(TypeError):
        ImbalanceSummary(5e-11, 2.5e-8, 1e-11)


@pytest.mark.unit
@pytest.mark.parametrize("position", [0, 1, 2, 3])
@pytest.mark.parametrize("bad", [0.0, -1.0, math.inf, math.nan, True])
def test_a_scale_or_tolerance_not_positive_and_finite_is_rejected(
    position: int, bad: float
) -> None:
    """Defect caught: the constructor's validation removed (zero, negative, NaN, inf, True)."""
    args = [SCALE, FLUX, 1e-6, 1e-10]
    args[position] = bad
    with pytest.raises(ValueError, match="must be positive and finite"):
        ErrorEstimateRule(*args)
