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
    viscosity_scale,
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
# Condition (e) and the rule's version (ADR-012 E, ECR-002 steps 6 and 7)
# ---------------------------------------------------------------------------

# A kinematic scale that is not 1, so a rule that forgot to divide by it fails.
NU_SCALE = 2.5e-3
# Air's kinematic viscosity on the Annex 20 room, and the room's median eddy
# viscosity over it on 216x72 (docs/reports/ecr002_val017_annex20.md, section
# 5.2): the scale (e) takes there since 2026-10-11, against the inlet's 2.22 nu.
NU_AIR = 15.3e-6
ROOM_SCALE = NU_AIR * (1.0 + 367.0)


def _rule_e() -> ErrorEstimateRule:
    """The rule at the default tolerances with condition (e) on."""
    return ErrorEstimateRule(SCALE, FLUX, 1e-6, 1e-10, viscosity_condition=True)


def _feed_e(
    rule: ErrorEstimateRule,
    nu_steps: np.ndarray,
    nu_scales: np.ndarray | float,
) -> list[bool]:
    """Feed (e) a history beside velocity steps that meet (a) from the window's end."""
    scales = np.broadcast_to(np.asarray(nu_scales, dtype=float), nu_steps.shape)
    velocity = 1e-12 * 0.5 ** np.arange(nu_steps.size)
    return [
        rule.update(
            float(s), lambda: _summary(), viscosity_step=float(e), nu_scale=float(n)
        )
        for s, e, n in zip(velocity, nu_steps, scales, strict=True)
    ]


def _first_stop(answers: list[bool]) -> int | None:
    """The index of the first True, or None when the history never stops."""
    return answers.index(True) if True in answers else None


@pytest.mark.unit
def test_the_version_is_the_rules() -> None:
    """3 without (e), 5 with it: a laminar solve records the conditions it applies."""
    assert _rule().version == 3
    assert _rule_e().version == 5


@pytest.mark.unit
@pytest.mark.parametrize("rho", [0.5, 0.9, 0.999])
def test_condition_e_is_the_estimate_on_the_eddy_viscosity(rho: float) -> None:
    """step_nu rho / (1 - rho) / nu_scale, fitted over the window as (a) is."""
    rule = _rule_e()
    nu_steps = _geometric(rho, RATE_WINDOW + 20, 1e-9)
    for nu_step in nu_steps:
        rule.update(
            1e-12, lambda: _summary(), viscosity_step=float(nu_step), nu_scale=NU_SCALE
        )
    expected = 1e-9 * rho / (1.0 - rho) / NU_SCALE
    assert rule.viscosity_estimate_history[-1] == pytest.approx(expected, rel=1e-9)
    assert len(rule.viscosity_estimate_history) == RATE_WINDOW + 20
    assert math.isinf(rule.viscosity_estimate_history[RATE_WINDOW - 2])
    assert rule.viscosity_scale_history == [NU_SCALE] * (RATE_WINDOW + 20)


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
        rule.update(float(s), summary, viscosity_step=float(e), nu_scale=NU_SCALE)
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
def test_a_room_sized_scale_stops_where_one_165_times_smaller_does_not() -> None:
    """The Annex 20 shape: the same contracting history, two scales.

    The viscosity steps contract at 0.998 per outer iteration, the rate
    prompt 49's records show, from a size at which the room's scale (nu
    plus its median nu_t, 368 nu) is met inside the history. The inlet's
    scale is 165 times smaller, and contracting the estimate by 165 at that
    rate takes about 2,550 more iterations than the history holds. Defect
    caught: the scale not applied, or applied as a product.
    """
    n = 3 * RATE_WINDOW
    rho = 0.998
    # The room's estimate crosses the tolerance half way from 150 to 151.
    start = 1e-6 * ROOM_SCALE * (1.0 - rho) / rho / rho**150.5
    nu_steps = start * rho ** np.arange(n)
    room = _feed_e(_rule_e(), nu_steps, ROOM_SCALE)
    small_rule = _rule_e()
    small = _feed_e(small_rule, nu_steps, ROOM_SCALE / 165.0)
    assert _first_stop(room) == 151
    assert _first_stop(small) is None
    assert small_rule.viscosity_estimate_history[-1] == pytest.approx(
        165.0 * 1e-6 * rho ** (n - 1 - 150.5), rel=1e-6
    )


@pytest.mark.unit
def test_the_scale_is_the_median_of_the_fluid_cells() -> None:
    """nu plus the median of nu_t over FLUID cells, whatever the other cells hold.

    Defects caught: the maximum in place of the median; the median over
    every cell; nu left out.
    """
    nu_t = np.array(
        [
            [9e9, 9e9, 9e9, 9e9],
            [9e9, 1.0e-3, 2.0e-3, 9e9],
            [9e9, 4.0e-3, 8.0e-3, 9e9],
            [9e9, 3.0e-3, 9e9, 9e9],
        ]
    )
    fluid = nu_t < 1.0
    assert viscosity_scale(NU_AIR, nu_t, fluid) == NU_AIR + 3.0e-3


@pytest.mark.unit
def test_one_cells_overshoot_leaves_the_stop_unchanged() -> None:
    """One FLUID cell's nu_t raised a thousandfold moves neither the scale nor the stop.

    The overshoot is the stagnation anomaly ADR-012 E names. The history
    contracts at 0.95 and stops at outer iteration 301 under the field's
    scale; a scale a thousand times larger, which the field's maximum would
    give, stops 135 iterations sooner. Defect caught: the maximum in place
    of the median.
    """
    rng = np.random.default_rng(50)
    nu_t = rng.uniform(1e-3, 2e-3, size=(12, 20))
    fluid = np.zeros_like(nu_t, dtype=bool)
    fluid[1:-1, 1:-1] = True
    spiked = nu_t.copy()
    top = np.unravel_index(np.argmax(np.where(fluid, nu_t, 0.0)), nu_t.shape)
    spiked[top] *= 1000.0
    base, overshoot = (viscosity_scale(NU_AIR, f, fluid) for f in (nu_t, spiked))
    assert overshoot == base
    n = 4 * RATE_WINDOW
    start = 1e-6 * base * (0.05 / 0.95) / 0.95**300.5
    nu_steps = start * 0.95 ** np.arange(n)
    stop = _first_stop(_feed_e(_rule_e(), nu_steps, base))
    assert stop is not None and stop > RATE_WINDOW + 150
    assert _first_stop(_feed_e(_rule_e(), nu_steps, overshoot)) == stop
    largest = NU_AIR + float(spiked[fluid].max())
    sooner = _first_stop(_feed_e(_rule_e(), nu_steps, largest))
    assert sooner is not None and sooner < stop - 100


@pytest.mark.unit
def test_each_step_is_estimated_over_the_scale_passed_with_it() -> None:
    """The scale of the step at hand, not the first one: the median moves with the iterate.

    The steps contract at 0.9 and the scale doubles at outer iteration 150;
    every estimate is the step's own over that step's scale. Defect caught:
    the first scale kept for the solve.
    """
    n = 2 * RATE_WINDOW
    nu_steps = _geometric(0.9, n, 1e-9)
    scales = np.where(np.arange(n) < 150, NU_SCALE, 2.0 * NU_SCALE)
    rule = _rule_e()
    _feed_e(rule, nu_steps, scales)
    estimates = np.array(rule.viscosity_estimate_history)[RATE_WINDOW - 1 :]
    expected = nu_steps[RATE_WINDOW - 1 :] * 9.0 / scales[RATE_WINDOW - 1 :]
    np.testing.assert_allclose(estimates, expected, rtol=1e-9)
    assert rule.viscosity_scale_history == list(scales)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("viscosity_step", "nu_scale"), [(None, None), (1e-6, None), (None, NU_SCALE)]
)
def test_condition_e_needs_the_step_and_its_scale(
    viscosity_step: float | None, nu_scale: float | None
) -> None:
    """Defect caught: (e) silently skipped when the step or the scale is missing."""
    with pytest.raises(ValueError, match="given exactly when the rule applies"):
        _rule_e().update(
            1e-3, lambda: _summary(), viscosity_step=viscosity_step, nu_scale=nu_scale
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("viscosity_step", "nu_scale"), [(1e-6, NU_SCALE), (1e-6, None), (None, NU_SCALE)]
)
def test_a_laminar_rule_refuses_a_viscosity_step_or_scale(
    viscosity_step: float | None, nu_scale: float | None
) -> None:
    """Defect caught: a step or a scale fed to a rule without (e), silently ignored."""
    with pytest.raises(ValueError, match="given exactly when the rule applies"):
        _rule().update(
            1e-3, lambda: _summary(), viscosity_step=viscosity_step, nu_scale=nu_scale
        )


@pytest.mark.unit
@pytest.mark.parametrize("bad", [0.0, -1.0, math.inf, math.nan, True])
def test_a_nu_scale_not_positive_and_finite_is_rejected(bad: float) -> None:
    """Defect caught: the per-step scale's validation removed; the step recorded first."""
    rule = _rule_e()
    with pytest.raises(ValueError, match="nu_scale must be positive and finite"):
        rule.update(1e-3, lambda: _summary(), viscosity_step=1e-6, nu_scale=bad)
    assert rule.estimate_history == []
    assert rule.viscosity_scale_history == []


@pytest.mark.unit
@pytest.mark.parametrize("bad", [1, "yes", None])
def test_viscosity_condition_must_be_a_bool(bad: object) -> None:
    """Defect caught: a truthy value turning (e) on, or a scale passed where the flag goes."""
    with pytest.raises(TypeError, match="viscosity_condition must be a bool"):
        ErrorEstimateRule(SCALE, FLUX, 1e-6, 1e-10, viscosity_condition=bad)  # type: ignore[arg-type]
