"""Unit tests for src/stopping.py on synthetic step histories.

Each test names the planted defect it was shown to catch.
"""

import dataclasses
import math

import numpy as np
import pytest

from src.stopping import (
    RATE_WINDOW,
    ErrorEstimateRule,
    ImbalanceSummary,
    IterationState,
)

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
def test_iteration_state_has_its_seven_fields_in_order() -> None:
    """The callback snapshot moved here from solver_ns.py with its fields unchanged.

    Defect caught: a field renamed, which every positional construction in
    the harness and the tests would still accept. The count field was renamed
    with the solve on 2026-10-06 (ECR-003 step 1); this pins the new name.
    """
    names = [field.name for field in dataclasses.fields(IterationState)]
    assert names == [
        "iteration",
        "residual",
        "pressure_iterations",
        "u",
        "v",
        "p",
        "pressure_products",
    ]
    u, v, p = np.zeros((2, 3)), np.ones((2, 3)), np.full((2, 3), 2.0)
    state = IterationState(4, 1e-3, 7, u, v, p, 8)
    assert (state.iteration, state.residual, state.pressure_iterations) == (4, 1e-3, 7)
    assert state.pressure_products == 8
    assert state.u is u and state.v is v and state.p is p
    # Frozen, and compared by identity only: the fields are arrays.
    with pytest.raises(dataclasses.FrozenInstanceError):
        state.iteration = 5  # type: ignore[misc]
    assert state != IterationState(4, 1e-3, 7, u, v, p, 8)


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


# ---------------------------------------------------------------------------
# Condition (e) and the rule's version (ADR-012 E, ECR-002 step 6)
# ---------------------------------------------------------------------------

# A kinematic scale that is not 1, so a rule that forgot to divide by it fails.
NU_SCALE = 2.5e-3


def _rule_e() -> ErrorEstimateRule:
    """The rule at the default tolerances with condition (e) on."""
    return ErrorEstimateRule(SCALE, FLUX, 1e-6, 1e-10, nu_scale=NU_SCALE)


@pytest.mark.unit
def test_the_version_is_the_rules() -> None:
    """3 without (e), 4 with it: a laminar solve records the conditions it applies."""
    assert _rule().version == 3
    assert _rule_e().version == 4


@pytest.mark.unit
@pytest.mark.parametrize("rho", [0.5, 0.9, 0.999])
def test_condition_e_is_the_estimate_on_the_eddy_viscosity(rho: float) -> None:
    """step_nu rho / (1 - rho) / nu_scale, fitted over the window as (a) is."""
    rule = _rule_e()
    nu_steps = _geometric(rho, RATE_WINDOW + 20, 1e-9)
    for nu_step in nu_steps:
        rule.update(1e-12, lambda: _summary(), viscosity_step=float(nu_step))
    expected = 1e-9 * rho / (1.0 - rho) / NU_SCALE
    assert rule.viscosity_estimate_history[-1] == pytest.approx(expected, rel=1e-9)
    assert len(rule.viscosity_estimate_history) == RATE_WINDOW + 20
    assert math.isinf(rule.viscosity_estimate_history[RATE_WINDOW - 2])


@pytest.mark.unit
def test_condition_e_holds_the_stop_until_the_viscosity_settles() -> None:
    """(a) to (d) met from the window's end; (e) decides the stop, then it stops.

    The velocity steps fall fast, the viscosity's slowly, so the stop is
    where (e)'s estimate first falls below the tolerance, and the imbalance
    is not asked for before that.
    """
    rule = _rule_e()
    n = 4 * RATE_WINDOW
    velocity = 1e-2 * 0.5 ** np.arange(n)
    viscosity = 1e-3 * 0.95 ** np.arange(n)
    asked = []

    def summary() -> ImbalanceSummary:
        asked.append(len(rule.estimate_history))
        return _summary()

    stops = [
        rule.update(float(s), summary, viscosity_step=float(e))
        for s, e in zip(velocity, viscosity, strict=True)
    ]
    first = stops.index(True)
    nu_estimates = np.array(rule.viscosity_estimate_history)
    assert first == int(np.argmax(nu_estimates < 1e-6))
    assert np.array(rule.estimate_history)[RATE_WINDOW - 1 : first].max() < 1e-6
    assert asked[0] == first + 1
    without = _rule()
    assert _feed(without, velocity).index(True) == RATE_WINDOW - 1 < first


@pytest.mark.unit
def test_the_viscosity_step_is_given_exactly_with_condition_e() -> None:
    """Defect caught: (e) silently skipped when the step is missing, or a step fed to no scale."""
    with pytest.raises(ValueError, match="viscosity_step is given exactly"):
        _rule_e().update(1e-3, lambda: _summary())
    with pytest.raises(ValueError, match="viscosity_step is given exactly"):
        _rule().update(1e-3, lambda: _summary(), viscosity_step=1e-6)


@pytest.mark.unit
@pytest.mark.parametrize("bad", [0.0, -1.0, math.inf, math.nan, True])
def test_a_nu_scale_not_positive_and_finite_is_rejected(bad: float) -> None:
    with pytest.raises(ValueError, match="nu_scale must be positive and finite"):
        ErrorEstimateRule(SCALE, FLUX, 1e-6, 1e-10, nu_scale=bad)
