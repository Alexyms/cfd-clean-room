"""Checks of prompt 85's probe: the variants change one thing, the comparisons see planted moves.

python -m pytest docs/reports/probe85/test_probe85.py -q
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import probe85  # noqa: E402 -- follows sys.path.insert
import tables85  # noqa: E402

from src import scalar_scheme  # noqa: E402


def _flatten(mapping: dict, prefix: str = "") -> dict:
    """Dotted keys to values, lists kept whole."""
    out = {}
    for key, value in mapping.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(_flatten(value, name + "."))
        else:
            out[name] = value
    return out


@pytest.mark.unit
@pytest.mark.parametrize("scenario", ["dust", "door", "breach"])
@pytest.mark.parametrize("variant", list(probe85.VARIANTS))
def test_a_variant_changes_only_its_key(variant: str, scenario: str) -> None:
    """Each variant's file differs from the reference's only where it says."""
    ref, _ = probe85.variant_mapping("reference", scenario, "80x30")
    var, changed = probe85.variant_mapping(variant, scenario, "80x30")
    a, b = _flatten(ref), _flatten(var)
    differ = {k for k in a.keys() | b.keys() if a.get(k) != b.get(k)}
    change = probe85.VARIANTS[variant]
    if change is None or change in ("sequential", "outer_later"):
        assert differ == set() and changed == []
    elif isinstance(change, tuple):
        assert differ == {f"{change[0]}.{change[1]}"}
        assert changed[0][2] == changed[0][1] * change[2]
    else:
        # The inflow: the room's supply always, the events' inlets inside the
        # events list, which is replaced whole.
        assert "boundaries.hepa_supply.velocity" in differ
        assert differ <= {"boundaries.hepa_supply.velocity", "scenario.events"}
        for _name, old, new in changed:
            assert new == old * probe85.INFLOW_FACTOR and new != old


@pytest.mark.unit
def test_the_grid_tolerance_is_prompt_79s_literal() -> None:
    """80x30's mass_imbalance_tol is configs79.PLACEMENT_TOL bit for bit."""
    assert probe85.mass_imbalance_tol("80x30") == 1.0e-4 * 1.2 * (0.1 * 0.1) / 60.0


@pytest.mark.unit
def test_relative_and_pointwise_see_a_planted_difference() -> None:
    """A planted change of 1e-9 at one cell reads as 1e-9 of the field's largest."""
    a = np.linspace(1.0, 10.0, 12).reshape(3, 4)
    b = a.copy()
    b[1, 2] += 1e-9
    assert tables85.relative(a, a.copy()) == 0.0
    assert tables85.relative(a, b) == pytest.approx(1e-9 / 10.0, rel=1e-6)
    assert tables85.pointwise(a, b, 0.0) == pytest.approx(1e-9 / a[1, 2], rel=1e-6)
    # Below the floor the pointwise measure does not look.
    assert tables85.pointwise(a, b, 100.0) == 0.0


@pytest.mark.unit
def test_moves_count_steps_and_one_only() -> None:
    """Times moved in steps; never against reached is one only; SOLID ignored."""
    plans = [
        {"t_start": 0.0, "dt": 0.1},
        {"t_start": 10.0, "dt": 0.2},
    ]
    ref = np.array([1.0, 11.0, math.inf, math.inf, math.nan, 5.0])
    var = np.array([1.1, 11.6, math.inf, 3.0, math.nan, 5.0 + 1e-12])
    steps = tables85.step_moves(ref, var, plans)
    assert steps[0] == pytest.approx(1.0)
    assert steps[1] == pytest.approx(3.0)  # the second interval's step
    assert steps[2] == 0.0
    assert math.isinf(steps[3])
    assert math.isnan(steps[4])
    n, one, more, only, largest = tables85.counts(steps)
    assert (n, one, more, only) == (5, 1, 1, 1)
    assert largest == pytest.approx(3.0)


@pytest.mark.unit
def test_interval_dt_puts_a_switch_time_in_the_interval_it_ends() -> None:
    """A time equal to an interval's start belongs to the one before (its last step)."""
    plans = [{"t_start": 0.0, "dt": 0.1}, {"t_start": 10.0, "dt": 0.2}]
    dts = tables85.interval_dt(plans, np.array([10.0, 10.2, 0.05]))
    assert list(dts) == [0.1, 0.2, 0.1]


@pytest.mark.unit
def test_sequential_leftover_sums_in_row_major_order() -> None:
    """The patched form is the left-to-right sum; the counts see a differing sum."""
    saved = (scalar_scheme._Acceptance._form, scalar_scheme._Acceptance.test)
    try:
        patch = probe85.SequentialLeftover()
        patch.install()
        # 1 first, then many terms each below half its last place: left to
        # right every one is lost, NumPy's pairwise sum adds them up first.
        residual = np.full((32, 32), 1e-16)
        residual[0, 0] = 1.0
        accept = scalar_scheme._Acceptance(1.0, np.zeros_like(residual, bool), 10.0)
        assert accept.test(residual)
        expected = 0.0
        for value in residual.ravel():
            expected += value
        pairwise = float(np.sum(residual))
        assert expected == 1.0 and pairwise > 1.0
        assert accept.leftover == expected
        assert patch.formed == 1 and patch.tests == 1 and patch.flipped == 0
        assert patch.differ == 1
        # A bound between the two sums flips the verdict.
        bound = (expected + pairwise) / 2.0
        accept = scalar_scheme._Acceptance(1.0, np.zeros_like(residual, bool), bound)
        assert accept.test(residual)
        assert patch.flipped == 1
    finally:
        scalar_scheme._Acceptance._form, scalar_scheme._Acceptance.test = saved


@pytest.mark.unit
def test_outer_one_later_defers_each_rules_first_stop_once() -> None:
    """The first True of each rule instance becomes False once; later answers pass."""
    from src.stopping import ErrorEstimateRule

    saved = ErrorEstimateRule.update
    try:
        later = probe85.OuterOneLater()
        later.original = lambda rule, *args, **kwargs: True
        later.install()
        first = object.__new__(ErrorEstimateRule)
        second = object.__new__(ErrorEstimateRule)
        assert [first.update(0.0, None) for _ in range(3)] == [False, True, True]
        assert [second.update(0.0, None) for _ in range(2)] == [False, True]
        assert later.deferred == 2
    finally:
        ErrorEstimateRule.update = saved
