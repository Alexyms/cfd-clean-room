"""The compiled core's stopping rule and outer iteration against the NumPy reference.

REQ-N03, ADR-017 D, G and H; prompt 95. ``_cfdcore.StoppingRule2`` against
``ErrorEstimateRule`` on synthetic histories: every verdict, window and scale bit
for bit, and each estimate to 1e-12 of the reference's, since its rate fit's
``x @ log(steps)`` is OpenBLAS's dot there and the core's fixed blocks here (the
second kind). ``viscosity_scale``'s median, the imbalance summary, the cell
centres and the max-norms bit for bit. Then ``tests/core_driver.CoreSolver``,
whose outer iterations run in a ``_cfdcore.Flow2``, against ``StaggeredSolver``:
outer iterations from planted fields on prompt 87's meshes, the cavity's and a
small room open on every edge, laminar, under a fixed eddy viscosity and coupled,
bit for bit with the reference's conjugate gradients given the core's dot product
(the chain judged block by block, each block handed what the other side's was), and
within level one's 1e-12 as the reference runs; the core's own restarts, callbacks
and divergence; whole solves at which the stop waits for continuity or falls between
the velocity step's bound and twice it, and the channel with its pressure outlet; the
failures a planted state reaches, compared with the reference's by type, message,
records, callbacks and log lines; and checkpoints that cross between the sides,
judged at level 2 (ADR-017 H).

The coupled blocks call the C library's logarithm and powers, so the tests that
compare them bit for bit ask for ``libm_numpy`` (prompt 94). Skipped with its
reason when ``_cfdcore`` cannot be imported, an error instead with
``CFD_CORE_REQUIRED=1``, as ``tests/test_core_parity.py``.
"""

import contextlib
import copy
import dataclasses
import functools
import logging
import math
import os
import re
from collections.abc import Callable, Iterator
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
import yaml

import src.time_integration as time_integration
from src import pressure, scalar_scheme
from src.boundary_staggered import StaggeredBoundary
from src.config import STANDARD, UPWIND, SimConfig
from src.io_manager import RunFile
from src.mesh import FLUID, SOLID, Mesh
from src.solver_staggered import CAP_STOP, FlowState, StaggeredSolver, SteadyCheckpoint
from src.staggered import allocate_fields, to_cell_centers
from src.stopping import (
    RATE_WINDOW,
    ErrorEstimateRule,
    ImbalanceSummary,
    IterationState,
    viscosity_scale,
)
from src.turbulence import PositivityError
from tests import core_mapping

if os.environ.get("CFD_CORE_REQUIRED") == "1":
    import _cfdcore
else:
    _cfdcore = pytest.importorskip(
        "_cfdcore",
        reason=(
            "the compiled core is not built: "
            "pip install --no-build-isolation ./core under WSL2 (ADR-017 A)"
        ),
    )

# After the skip above: these modules need the core too.
from tests.core_driver import CoreSolver, steady_solves_in_the_core
from tests.test_core_parity import MESHES, _raw_config
from tests.test_core_turbulence import _raw as _turbulence_raw
from tests.test_core_turbulence import (
    _small_raw,
    libm_numpy,  # noqa: F401 -- a fixture, used by name
)

pytestmark = [pytest.mark.parity, pytest.mark.integration]

SEED = 95
# Level one's bound on a value of the second kind (REQ-N03).
LEVEL_ONE = 1e-12
# Level two's limit on a flow field (ADR-017 H, item 2).
FLOW_LIMIT = 1e-8
MODES = ["laminar", "fixed", "coupled"]
ROOM = "40x15-uniform-open"
CAVITY = 16
# The small room of the turbulence tests, a pressure outlet on every edge, on 12 by 7
# cells: the one room of the outer-iteration cases whose outlets are read inside it.
SMALL = "small-outlets"
SMALL_SIZE = (12, 7)


def _bits(a: object) -> np.ndarray:
    """A float or an array as its 64-bit patterns."""
    return np.ascontiguousarray(np.asarray(a, dtype=np.float64)).view(np.uint64)


def _same(a: object, b: object) -> bool:
    """Whether two floats or arrays are equal bit for bit."""
    x, y = _bits(a), _bits(b)
    return x.shape == y.shape and bool(np.array_equal(x, y))


def _rng(*parts: object) -> np.random.Generator:
    """A generator seeded by the case's own parts."""
    return np.random.default_rng([SEED, *(sum(map(ord, str(p))) for p in parts)])


def _relative(a: float, b: float) -> float:
    """|a - b| / |b|; zero for two equal values, infinities included."""
    if a == b:
        return 0.0
    return abs(a - b) / abs(b)


# ---- The rule on synthetic histories ---------------------------------------------


def _pair(
    viscosity_condition: bool = False,
    velocity_scale: float = 0.45,
    flux_scale: float = 1.3,
    error_tol: float = 1e-6,
    imbalance_tol: float = 1e-9,
) -> tuple[ErrorEstimateRule, "_cfdcore.StoppingRule2"]:
    """The reference's rule and the core's, from the same arguments."""
    arguments = (velocity_scale, flux_scale, error_tol, imbalance_tol)
    keys = {"viscosity_condition": viscosity_condition}
    return (
        ErrorEstimateRule(*arguments, **keys),
        core_mapping.error_estimate_rule(*arguments, **keys),
    )


class Imbalance:
    """An imbalance reading that counts its calls."""

    def __init__(self, worst: float, absolute_sum: float, signed_sum: float) -> None:
        self.summary = ImbalanceSummary(
            worst=worst, absolute_sum=absolute_sum, signed_sum=signed_sum
        )
        self.calls = 0

    def __call__(self) -> ImbalanceSummary:
        self.calls += 1
        return self.summary


PASSING = (1e-10, 1e-7, -5e-10)


def _feed(
    rules: tuple[ErrorEstimateRule, "_cfdcore.StoppingRule2"],
    steps: list[float],
    viscosity: list[float] | None = None,
    scales: list[float] | None = None,
    reading: tuple[float, float, float] = PASSING,
) -> tuple[list[bool], list[bool], list[int]]:
    """Both rules fed the same steps: each one's verdicts, and the calls each made."""
    ref, core = rules
    verdicts: tuple[list[bool], list[bool]] = ([], [])
    calls = []
    for n, step in enumerate(steps):
        keys = {}
        if viscosity is not None and scales is not None:
            keys = {"viscosity_step": viscosity[n], "nu_scale": scales[n]}
        counted = [Imbalance(*reading), Imbalance(*reading)]
        verdicts[0].append(ref.update(step, counted[0], **keys))
        verdicts[1].append(core.update(step, counted[1], **keys))
        assert counted[0].calls == counted[1].calls
        calls.append(counted[0].calls)
    return verdicts[0], verdicts[1], calls


def _assert_estimates(
    expected: list[float], got: list[float], steps: list[float], scales: list[float]
) -> float:
    """The estimates infinite at the same places, and otherwise level one on the rate.

    An estimate is ``step rho / (1 - rho) / scale``: a relative change of the fitted
    rate's logarithm moves it by ``1 / (1 - rho)`` times as much, so a dot product
    of the second kind that is level one's below 1e-12 can move an estimate of a
    rate near one by more. The bound is 1e-12 times that factor, the reference's
    rho read from its own estimate: ``1 / (1 - rho) = 1 + estimate scale / step``.
    Returns the largest relative difference over that factor.
    """
    assert len(expected) == len(got) == len(steps) == len(scales)
    worst = 0.0
    for a, b, step, scale in zip(got, expected, steps, scales, strict=True):
        assert math.isinf(a) == math.isinf(b), (a, b)
        if math.isinf(b):
            continue
        factor = 1.0 + b * scale / step
        worst = max(worst, _relative(a, b) / factor)
    assert worst <= LEVEL_ONE
    return worst


def _geometric(rate: float, n: int, first: float = 1e-3) -> list[float]:
    """Steps first * rate^k, with a ripple of a part in 1e4 that the fit averages."""
    rng = _rng("ripple", rate, n)
    return [first * rate**k * (1.0 + 1e-4 * rng.standard_normal()) for k in range(n)]


@pytest.mark.parametrize("rate", [0.5, 0.97, 0.9999, 1.0, 1.0001, 1.05])
def test_a_geometric_history_reads_as_the_references(rate: float) -> None:
    """The window filling, the estimate of each rate, a rate at and beyond 1 inf."""
    rules = _pair()
    steps = _geometric(rate, 2 * RATE_WINDOW + 17)
    ref_verdicts, core_verdicts, _ = _feed(rules, steps)
    assert core_verdicts == ref_verdicts
    ref, core = rules
    _assert_estimates(
        ref.estimate_history, core.estimate_history, steps, [0.45] * len(steps)
    )
    estimates = core.estimate_history
    assert all(math.isinf(e) for e in estimates[: RATE_WINDOW - 1])
    if rate < 1.0:
        assert all(math.isfinite(e) for e in estimates[RATE_WINDOW - 1 :])
    if rate >= 1.0001:
        assert all(math.isinf(e) for e in estimates)
    assert core.window() == ref.window()
    assert len(core.window()[0]) == RATE_WINDOW


def test_a_constant_history_has_its_rate_at_one() -> None:
    """Equal steps fit a slope of 0, a rate of exactly 1: no estimate."""
    rules = _pair()
    _feed(rules, [1e-4] * (RATE_WINDOW + 5))
    assert all(math.isinf(e) for e in rules[1].estimate_history)
    assert rules[1].estimate_history == rules[0].estimate_history


@pytest.mark.parametrize("planted", [0.0, -1e-7, math.nan, math.inf])
def test_a_step_that_is_zero_or_not_finite_leaves_no_estimate_while_in_the_window(
    planted: float,
) -> None:
    """The plant reads inf until it leaves the window, then the estimates resume."""
    rules = _pair()
    steps = _geometric(0.95, 3 * RATE_WINDOW)
    at = RATE_WINDOW + 20
    steps[at] = planted
    ref_verdicts, core_verdicts, _ = _feed(rules, steps)
    assert core_verdicts == ref_verdicts
    ref, core = rules
    _assert_estimates(
        ref.estimate_history, core.estimate_history, steps, [0.45] * len(steps)
    )
    for n, estimate in enumerate(core.estimate_history):
        held = at <= n < at + RATE_WINDOW
        assert math.isinf(estimate) == (held or n < RATE_WINDOW - 1), n


def _below_tolerance_history() -> list[float]:
    """Steps whose estimate falls below 1e-6 of the scale once the window fills."""
    return _geometric(0.5, RATE_WINDOW + 3, first=1e-6)


@pytest.mark.parametrize(
    ("reading", "accepts"),
    [
        (PASSING, True),
        ((2e-9, 1e-7, -5e-10), False),  # (b): the worst cell
        ((1e-10, 2e-6, -5e-10), False),  # (c): the summed magnitude over the flux
        ((1e-10, 1e-7, -2e-9), False),  # (d): the signed sum, in absolute value
        ((1e-10, 1e-7, 2e-9), False),
        ((1e-10, 1.29e-6, 5e-10), True),  # (c) just below: 1.29e-6 / 1.3 < 1e-6
    ],
)
def test_each_continuity_condition_fails_alone(
    reading: tuple[float, float, float], accepts: bool
) -> None:
    """(b), (c) and (d) each refuse the stop alone; the imbalance is read once (a) holds."""
    rules = _pair()
    ref_verdicts, core_verdicts, calls = _feed(
        rules, _below_tolerance_history(), reading=reading
    )
    assert core_verdicts == ref_verdicts
    assert core_verdicts[-1] is accepts
    assert calls[: RATE_WINDOW - 1] == [0] * (RATE_WINDOW - 1)
    assert calls[-1] == 1


def test_condition_a_fails_alone() -> None:
    """An estimate above the tolerance: no stop, and the imbalance is never read."""
    rules = _pair(error_tol=1e-12)
    ref_verdicts, core_verdicts, calls = _feed(rules, _geometric(0.99, RATE_WINDOW + 3))
    assert core_verdicts == ref_verdicts == [False] * len(calls)
    assert sum(calls) == 0


@pytest.mark.parametrize("fails", ["none", "e", "a"])
def test_condition_e_is_read_with_its_own_scale(fails: str) -> None:
    """(e) on: its estimate over the scale handed with each step, and the version 5."""
    rules = _pair(viscosity_condition=True)
    ref, core = rules
    assert ref.version == core.version == 5
    n = RATE_WINDOW + 3
    steps = _below_tolerance_history() if fails != "a" else _geometric(0.99, n)
    viscosity = _geometric(0.5 if fails != "e" else 0.999, n, first=1e-9)
    scales = [1.5e-5 * (1.0 + 0.01 * k) for k in range(n)]
    ref_verdicts, core_verdicts, calls = _feed(rules, steps, viscosity, scales)
    assert core_verdicts == ref_verdicts
    assert core_verdicts[-1] is (fails == "none")
    assert (calls[-1] == 1) is (fails == "none")
    _assert_estimates(ref.estimate_history, core.estimate_history, steps, [0.45] * n)
    _assert_estimates(
        ref.viscosity_estimate_history,
        core.viscosity_estimate_history,
        viscosity,
        scales,
    )
    assert _same(core.viscosity_scale_history, ref.viscosity_scale_history)
    assert core.window() == ref.window()


def test_condition_e_off_is_version_3_with_no_viscosity_readings() -> None:
    """(e) off: the version 3, and the viscosity's histories and window empty."""
    ref, core = _pair()
    assert ref.version == core.version == 3
    _feed((ref, core), _below_tolerance_history())
    assert core.viscosity_estimate_history == core.viscosity_scale_history == []
    assert core.window()[1] == ()


@pytest.mark.parametrize("on", [False, True])
def test_a_restored_rule_continues_the_one_it_was_taken_from(on: bool) -> None:
    """A window restored into a new rule: its next verdicts and estimates are the old one's."""
    rules = _pair(viscosity_condition=on)
    n = 2 * RATE_WINDOW + 40
    steps = _geometric(0.9, n, first=1e-3)
    viscosity = _geometric(0.9, n, first=1e-6) if on else None
    scales = [2e-5] * n if on else None
    split = RATE_WINDOW + 50
    _feed(
        rules,
        steps[:split],
        None if viscosity is None else viscosity[:split],
        None if scales is None else scales[:split],
    )
    window = rules[1].window()
    assert window == rules[0].window()
    fresh = _pair(viscosity_condition=on)
    fresh[0].restore(*window)
    fresh[1].restore(*window)
    rest = (
        steps[split:],
        None if viscosity is None else viscosity[split:],
        None if scales is None else scales[split:],
    )
    continued = _feed(rules, *rest)
    restored = _feed(fresh, *rest)
    assert continued[1] == restored[1] == restored[0]
    assert _same(rules[1].estimate_history[split:], fresh[1].estimate_history), (
        "the core's restored rule departs from the core's own"
    )
    _assert_estimates(
        fresh[0].estimate_history,
        fresh[1].estimate_history,
        steps[split:],
        [0.45] * len(steps[split:]),
    )


@pytest.mark.parametrize(
    ("arguments", "keys", "error"),
    [
        ((0.0, 1.0, 1e-6, 1e-9), {}, ValueError),
        ((True, 1.0, 1e-6, 1e-9), {}, ValueError),
        ((0.4, math.nan, 1e-6, 1e-9), {}, ValueError),
        ((0.4, 1.0, -1e-6, 1e-9), {}, ValueError),
        ((0.4, 1.0, 1e-6, math.inf), {}, ValueError),
        ((0.4, 1.0, 1e-6, "1e-9"), {}, TypeError),
        ((0.4, 1.0, 1e-6, 1e-9), {"viscosity_condition": 1}, TypeError),
    ],
)
def test_the_rules_construction_is_refused_as_the_reference_refuses_it(
    arguments: tuple, keys: dict, error: type[Exception]
) -> None:
    """Each scale and tolerance positive and finite, a bool refused; the condition a bool."""
    with pytest.raises(error) as expected:
        ErrorEstimateRule(*arguments, **keys)
    with pytest.raises(error) as got:
        core_mapping.error_estimate_rule(*arguments, **keys)
    assert str(got.value) == str(expected.value)


@pytest.mark.parametrize(
    ("on", "keys"),
    [
        (False, {"viscosity_step": 1e-6, "nu_scale": 1e-5}),
        (False, {"viscosity_step": 1e-6}),
        (False, {"nu_scale": 1e-5}),
        (True, {}),
        (True, {"viscosity_step": 1e-6}),
        (True, {"viscosity_step": 1e-6, "nu_scale": 0.0}),
        (True, {"viscosity_step": 1e-6, "nu_scale": math.nan}),
        (True, {"viscosity_step": 1e-6, "nu_scale": True}),
    ],
)
def test_update_is_refused_as_the_reference_refuses_it(on: bool, keys: dict) -> None:
    """The viscosity's step and scale given exactly with (e), the scale checked; no trace left."""
    rules = _pair(viscosity_condition=on)
    messages = []
    for rule in rules:
        with pytest.raises(ValueError) as raised:
            rule.update(1e-3, Imbalance(*PASSING), **keys)
        messages.append(str(raised.value))
    assert messages[1] == messages[0]
    assert rules[1].estimate_history == rules[0].estimate_history == []
    assert rules[1].window() == rules[0].window() == ((), ())


@pytest.mark.parametrize(
    ("on", "taken", "steps", "viscosity"),
    [
        (False, True, (1e-3,), ()),
        (False, False, tuple([1e-3] * (RATE_WINDOW + 1)), ()),
        (False, False, (1e-3, 1), ()),
        (False, False, (1e-3, True), ()),
        (False, False, (1e-3, math.nan), ()),
        (False, False, (1e-3, -1e-3), ()),
        (False, False, (1e-3,), (1e-6,)),
        (True, False, (1e-3,), ()),
        (True, False, (1e-3,), (1e-6, 1e-6)),
        (True, False, (1e-3,), (math.inf,)),
    ],
)
def test_restore_is_refused_as_the_reference_refuses_it(
    on: bool, taken: bool, steps: tuple, viscosity: tuple
) -> None:
    """A rule that has stepped, a window too long, a value not a finite float, a count off."""
    rules = _pair(viscosity_condition=on)
    messages = []
    for rule in rules:
        if taken:
            keys = {"viscosity_step": 1e-6, "nu_scale": 1e-5} if on else {}
            rule.update(1e-3, Imbalance(*PASSING), **keys)
        with pytest.raises(ValueError) as raised:
            rule.restore(steps, viscosity)
        messages.append(str(raised.value))
        assert rule.window()[0] == ((1e-3,) if taken else ())
    assert messages[1] == messages[0]


def test_the_rate_fits_x_dot_x_is_exact_at_the_window() -> None:
    """``x @ x`` over the centred positions: every partial sum is a whole number of quarters."""
    x = np.arange(RATE_WINDOW) - (RATE_WINDOW - 1) / 2.0
    exact = sum((Fraction(value) ** 2 for value in x), Fraction(0))
    assert Fraction(float(x @ x)) == exact
    total = Fraction(0)
    for value in x:
        total += Fraction(value) ** 2
        assert (total * 4).denominator == 1 and total < 2**51


def test_planted_windows_are_estimated_within_level_one() -> None:
    """Windows over decades, by ``_estimate_of``: the second kind, to 1e-12 relative."""
    _, core = _pair()
    rng = _rng("planted windows")
    worst = 0.0
    finite = 0
    for _ in range(400):
        rate = rng.uniform(0.3, 1.02)
        steps = [
            10.0 ** rng.uniform(-9.0, 0.0) * rate**k * (1.0 + 0.3 * rng.random())
            for k in range(RATE_WINDOW)
        ]
        expected = ErrorEstimateRule._estimate_of(steps, 0.45)  # type: ignore[arg-type]
        got = core.estimate_of(steps, 0.45)
        finite += math.isfinite(expected)
        worst = max(worst, _assert_estimates([expected], [got], [steps[-1]], [0.45]))
    assert worst <= LEVEL_ONE
    assert finite > 100


# ---- The median, the imbalance, the centres and the max-norms --------------------


@pytest.mark.parametrize("count", [1, 2, 3, 4, 7, 8, 1001, 1002])
@pytest.mark.parametrize("kind", ["spread", "ties", "zeros", "infinite", "nan"])
def test_the_median_is_numpys(count: int, kind: str) -> None:
    """``viscosity_scale`` on odd and even counts, ties, zeros, an infinity and a NaN."""
    rng = _rng("median", count, kind)
    shape = (7, 300)
    nu_t = 10.0 ** rng.uniform(-8.0, -2.0, shape)
    fluid = np.zeros(shape, dtype=bool)
    fluid.flat[rng.choice(nu_t.size, size=count, replace=False)] = True
    if kind == "ties":
        nu_t = np.round(rng.uniform(0.0, 3.0, shape)) * 1e-4
    elif kind == "zeros":
        nu_t[rng.random(shape) < 0.6] = 0.0
    elif kind == "infinite":
        nu_t.flat[np.flatnonzero(fluid)[0]] = math.inf
    elif kind == "nan":
        nu_t.flat[np.flatnonzero(fluid)[-1]] = math.nan
    nu = 1.5e-5
    expected = viscosity_scale(nu, nu_t, fluid)
    got = _cfdcore.viscosity_scale(nu, nu_t, fluid)
    assert _same(got, expected)


def _faces(mesh: Mesh, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    ny, nx = mesh.cell_type.shape
    return rng.standard_normal((ny, nx + 1)), rng.standard_normal((ny + 1, nx))


@pytest.mark.parametrize("name", list(MESHES))
def test_the_imbalance_summary_is_the_solvers(name: str) -> None:
    """``_imbalance_summary`` of planted faces, the sums in NumPy's order on every size.

    160x60 and 320x120 hold more cells than NumPy's buffer, so a sum read through
    it would part from the pairwise sum of the whole field there.
    """
    config = SimConfig.from_dict(_raw_config(name), "configs")
    mesh = Mesh(config)
    solver = StaggeredSolver(mesh, config, StaggeredBoundary(mesh, config))
    rng = _rng("imbalance", name)
    for nan in (False, True):
        u, v = _faces(mesh, rng)
        if nan:
            u[3, 4] = math.nan
        expected = solver._imbalance_summary(u, v)
        imbalance = solver._corrector.mass_imbalance(u, v)
        worst, absolute, signed = _cfdcore.imbalance_summary(imbalance)
        assert _same(worst, expected.worst)
        assert _same(absolute, expected.absolute_sum)
        assert _same(signed, expected.signed_sum)


@pytest.mark.parametrize("name", list(MESHES))
def test_the_cell_centres_and_their_changes_are_the_references(name: str) -> None:
    """``to_cell_centers`` and ``np.max(np.abs(a[m] - b[m]))``, a NaN read and one not read."""
    config = SimConfig.from_dict(_raw_config(name), "configs")
    mesh = Mesh(config)
    grid = core_mapping.core_grid(mesh)
    rng = _rng("centres", name)
    fluid = mesh.cell_type == FLUID
    u, v = _faces(mesh, rng)
    expected = to_cell_centers(u, v)
    got = _cfdcore.cell_centres(grid, u, v)
    assert _same(got[0], expected[0]) and _same(got[1], expected[1])
    before = rng.standard_normal(fluid.shape)
    for planted in (None, "read", "unread"):
        now = np.array(expected[0])
        if planted == "read":
            now.flat[np.flatnonzero(fluid)[5]] = math.nan
        elif planted == "unread":
            now.flat[np.flatnonzero(~fluid)[0]] = math.nan
        reference = float(np.max(np.abs(now[fluid] - before[fluid])))
        assert _same(_cfdcore.largest_change(now, before, fluid), reference)


@pytest.mark.parametrize(
    "values",
    [
        [1.0, 2.0],
        [2.0, 1.0],
        [math.nan, 1.0],
        [1.0, math.nan],
        [math.nan, math.nan],
        [-0.0, 0.0],
        [0.0, -0.0],
        [3.0, math.nan, 4.0],
        [math.inf, math.nan],
    ],
)
def test_the_velocity_steps_max_is_pythons(values: list[float]) -> None:
    """``max(du, dv)``: a NaN first is kept and a NaN later is not, a tie keeps the first."""
    assert _same(_cfdcore.python_max(values), max(values))


# ---- Outer iterations from planted fields ----------------------------------------


def _cavity_raw(mode: str) -> dict:
    """The cavity at CAVITY cells a side; coupled, the turbulence tests' moving walls."""
    if mode == "coupled":
        raw = _turbulence_raw("cavity")
    else:
        with open("configs/validation_cavity.yaml", encoding="utf-8") as stream:
            raw = yaml.safe_load(stream)
    raw = copy.deepcopy(raw)
    raw["domain"].update(nx=CAVITY, ny=CAVITY)
    return raw


def _raw(name: str, mode: str, **solver: object) -> dict:
    """A mesh's configuration: laminar or fixed without the turbulence section."""
    if name == "cavity":
        raw = _cavity_raw(mode)
    elif name == SMALL:
        raw = _small_raw(*SMALL_SIZE, 1.05, STANDARD, UPWIND)
    else:
        raw = copy.deepcopy(_raw_config(name))
    if mode != "coupled":
        # The inlets' turbulence keys are refused without the section that reads them.
        raw.pop("turbulence", None)
        for spec in raw.get("boundaries", {}).values():
            spec.pop("turbulence_intensity", None)
            spec.pop("dissipation_length", None)
    raw.setdefault("solver", {}).update(solver)
    return raw


def _configured(raw: dict) -> tuple[Mesh, SimConfig]:
    config = SimConfig.from_dict(raw, "configs")
    return Mesh(config), config


def _solver(kind: type, mesh: Mesh, config: SimConfig) -> StaggeredSolver:
    return kind(mesh, config, StaggeredBoundary(mesh, config))


def _planted(name: str, mode: str, rng: np.random.Generator) -> FlowState:
    """A state three outer iterations from rest, each field scaled by a planted ripple."""
    mesh, config = _configured(_raw(name, mode, max_simple_iter=3))
    solver = _solver(StaggeredSolver, mesh, config)
    solver.solve_steady()
    state = solver.flow_state
    assert state is not None

    def rippled(a: np.ndarray, size: float) -> np.ndarray:
        return a * (1.0 + size * rng.uniform(-1.0, 1.0, a.shape))

    turbulent = {}
    if state.k is not None:
        turbulent = {
            "k": rippled(state.k, 0.2),
            "eps": rippled(state.eps, 0.2),
            "nu_t": rippled(state.nu_t, 0.2),
        }
    return FlowState(
        u=rippled(state.u, 0.05),
        v=rippled(state.v, 0.05),
        p=rippled(state.p, 0.05),
        **turbulent,
    )


def _eddy_viscosity(mesh: Mesh, rng: np.random.Generator) -> np.ndarray:
    """A planted kinematic eddy viscosity, zero in SOLID cells."""
    shape = mesh.cell_type.shape
    return np.where(
        mesh.cell_type == SOLID, 0.0, 10.0 ** rng.uniform(-5.0, -3.0, shape)
    )


class _NumpyWithTheCoresDot:
    """NumPy, with ``vdot`` the core's row-blocked dot product (``_cfdcore.row_dot``)."""

    def __getattr__(self, name: str) -> object:
        return getattr(np, name)

    @staticmethod
    def vdot(a: np.ndarray, b: np.ndarray) -> float:
        """The core's dot product of two [ny, nx] arrays, a block per row."""
        return _cfdcore.row_dot(a, b)


@contextlib.contextmanager
def _core_dots() -> Iterator[None]:
    """The reference's conjugate gradients with the core's dot product (prompts 88, 93)."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(pressure, "np", _NumpyWithTheCoresDot())
        patch.setattr(scalar_scheme, "np", _NumpyWithTheCoresDot())
        yield


class _RuleInputs:
    """Keeps what each ``ErrorEstimateRule.update`` of the reference is handed."""

    def __init__(self) -> None:
        self.kept: dict[str, list[float]] = {
            "step": [],
            "viscosity_step": [],
            "nu_scale": [],
        }

    @contextlib.contextmanager
    def watching(self) -> Iterator[None]:
        """Keep the arguments of each ``ErrorEstimateRule.update`` while the block runs."""
        original = ErrorEstimateRule.update
        kept = self.kept

        def update(
            rule: ErrorEstimateRule,
            step: float,
            imbalance: object,
            viscosity_step: float | None = None,
            nu_scale: float | None = None,
        ) -> bool:
            kept["step"].append(step)
            kept["viscosity_step"].append(
                math.nan if viscosity_step is None else viscosity_step
            )
            kept["nu_scale"].append(math.nan if nu_scale is None else nu_scale)
            return original(rule, step, imbalance, viscosity_step, nu_scale)  # type: ignore[arg-type]

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(ErrorEstimateRule, "update", update)
            yield


def _pair_of_solves(
    name: str,
    mode: str,
    outer: int,
    *,
    dots: bool,
    inputs: _RuleInputs | None = None,
    **solver_keys: object,
) -> tuple[StaggeredSolver, CoreSolver]:
    """OUTER iterations warm-started from a planted state, the reference's and the core's."""
    rng = _rng("outer", name, mode, outer)
    initial = _planted(name, mode, rng)
    keys: dict[str, object] = {"max_simple_iter": outer, **solver_keys}
    if dots:
        keys["pressure_preconditioner"] = "jacobi"
    mesh, config = _configured(_raw(name, mode, **keys))
    extra: dict[str, object] = {"initial": initial}
    if mode == "fixed":
        extra["eddy_viscosity"] = _eddy_viscosity(mesh, rng)
    solved = []
    for kind in (StaggeredSolver, CoreSolver):
        solver = _solver(kind, mesh, config)
        context = _core_dots() if dots else contextlib.nullcontext()
        watch = (
            inputs.watching()
            if inputs is not None and kind is StaggeredSolver
            else contextlib.nullcontext()
        )
        with context, watch, np.errstate(all="ignore"):
            solver.solve_steady(**extra)
        solved.append(solver)
    return solved[0], solved[1]


FIELDS = ("u", "v", "p", "k", "eps", "nu_t")
OUTER_CASES = [
    (name, mode)
    for name in [*MESHES, "cavity", SMALL]
    for mode in MODES
    # The cavity's walls admit no air, which the coupled solve starts from.
    if not (name == "320x120" and mode != "coupled")
    and (name, mode) != ("cavity", "coupled")
]


@pytest.mark.usefixtures("libm_numpy")
@pytest.mark.parametrize(("name", "mode"), OUTER_CASES)
def test_outer_iterations_are_the_references_block_by_block(
    name: str, mode: str
) -> None:
    """Three outer iterations bit for bit, each block handed what the reference's was.

    The reference's conjugate gradients take the core's dot product and both sides
    the diagonal preconditioner, so that each block of the chain is given the same
    inputs on both sides: the outlets, the prediction, the correction, the
    k-epsilon step and the relaxation, the cell centres, the residual and the rule.
    """
    inputs = _RuleInputs()
    ref, core = _pair_of_solves(name, mode, 3, dots=True, inputs=inputs)
    assert _same(core.residual_history, ref.residual_history)
    # The rule's inputs: the velocity step, and with the model the viscosity's step
    # and its scale, the median of nu_t over the FLUID cells.
    for reading, kept in inputs.kept.items():
        assert len(kept) == 3
        assert _same(core.rule_readings[reading], kept), reading
    assert core.last_pressure_iterations == ref.last_pressure_iterations
    assert core.pressure_cap_hits == ref.pressure_cap_hits
    assert core.stop_reason == ref.stop_reason == CAP_STOP
    assert ref.flow_state is not None and core.flow_state is not None
    for field in FIELDS:
        expected = getattr(ref.flow_state, field)
        got = getattr(core.flow_state, field)
        assert (
            (got is None)
            == (expected is None)
            == (field in FIELDS[3:] and mode != "coupled")
        )
        if expected is not None:
            assert _same(got, expected), field
    assert _same(core.last_mass_imbalance, ref.last_mass_imbalance)


# The small room's flow grows without bound over its iterations (its fields reach
# 1e30 in three), where a difference of a part in 1e12 at the first is no
# comparison; its block-by-block cases are bit for bit.
@pytest.mark.parametrize(
    ("name", "mode"), [(n, m) for n, m in OUTER_CASES if n not in ("320x120", SMALL)]
)
def test_outer_iterations_are_within_level_one_as_the_reference_runs(
    name: str, mode: str
) -> None:
    """The same iterations as the reference runs them, its dots OpenBLAS's: 1e-12 per field.

    The correction is a chain whose own differences a later block can amplify
    (ADR-017 G), so the field bound is level one's on the outer iterations' result,
    with the conjugate gradients' counts equal.
    """
    ref, core = _pair_of_solves(name, mode, 3, dots=False)
    assert core.last_pressure_iterations == ref.last_pressure_iterations
    for field in FIELDS:
        expected = getattr(ref.flow_state, field)
        if expected is None:
            continue
        got = getattr(core.flow_state, field)
        scale = float(np.max(np.abs(expected)))
        assert float(np.max(np.abs(got - expected))) <= LEVEL_ONE * scale, field
    for a, b in zip(core.residual_history, ref.residual_history, strict=True):
        assert _relative(a, b) <= 1e-9


@pytest.mark.parametrize("name", [*MESHES, "80x30-vents-shelf"])
def test_the_effective_viscosity_is_the_solvers(name: str) -> None:
    """``_effective_viscosity`` of a planted nu_t, a NaN in a SOLID cell not read; its refusals."""
    from tests.test_core_parity import _raw_config as raw_of

    mesh, config = _configured(raw_of(name))
    solver = _solver(StaggeredSolver, mesh, config)
    live = mesh.cell_type != SOLID
    rng = _rng("effective", name)
    nu_t = 10.0 ** rng.uniform(-7.0, -2.0, live.shape)
    nu_t[rng.random(live.shape) < 0.1] = 0.0
    if (~live).any():
        nu_t[~live] = math.nan
    got = _cfdcore.effective_viscosity(nu_t, live, config.mu, config.rho)
    assert _same(got, solver._effective_viscosity(nu_t))
    for planted, words in ((math.inf, "finite"), (-1e-9, "non-negative")):
        spoilt = np.array(nu_t)
        spoilt.flat[np.flatnonzero(live)[3]] = planted
        with pytest.raises(ValueError) as expected:
            solver._effective_viscosity(spoilt)
        with pytest.raises(ValueError, match=words) as raised:
            _cfdcore.effective_viscosity(spoilt, live, config.mu, config.rho)
        assert str(raised.value) == str(expected.value)


@pytest.mark.parametrize(("name", "mode"), [(ROOM, "coupled"), ("cavity", "laminar")])
def test_the_rule_is_built_with_the_solvers_scales(name: str, mode: str) -> None:
    """The core's rule reads the solver's velocity scale and flux scale, and the window's length."""
    mesh, config = _configured(_raw(name, mode))
    solver = _solver(CoreSolver, mesh, config)
    core = core_mapping.flow_solver(
        solver._modules,
        config,
        solver._boundary,
        solver.reference_velocity,
        solver.flux_scale,
    )
    assert core.rule_keys == (
        RATE_WINDOW,
        solver._boundary.get_max_boundary_velocity(),
        solver.flux_scale,
        config.iteration_error_tol,
        config.mass_imbalance_tol,
        mode == "coupled",
    )
    assert core.rule_version == solver.rule_version


def test_a_capped_correction_does_not_stop_the_velocity_step(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Under velocity_step a step below the bound with its correction at the cap goes on.

    Every correction stops at its cap: the warning is logged at the first alone, as
    the reference logs it.
    """
    raw = _raw(
        "cavity",
        "laminar",
        stopping_rule="velocity_step",
        convergence_tol=0.5,
        max_pressure_iter=2,
        max_simple_iter=40,
    )
    solved = []
    lines = []
    for kind in (StaggeredSolver, CoreSolver):
        caplog.clear()
        with caplog.at_level(logging.INFO, logger="src.solver_staggered"):
            solved.append(_whole(kind, raw)[0])
        lines.append(_log_lines(caplog.records))
    assert lines[1] == lines[0]
    assert (
        sum(
            level == "WARNING" and "max_pressure_iter (" in text
            for level, text in lines[1]
        )
        == 1
    )
    ref, core = solved
    assert ref.stop_reason == core.stop_reason == CAP_STOP
    assert len(core.residual_history) == len(ref.residual_history) == 40
    assert min(core.residual_history) < 0.5
    assert core.pressure_cap_hits == ref.pressure_cap_hits == 40
    # The same bound with the cap out of reach stops at once.
    raw["solver"]["max_pressure_iter"] = 5000
    stopped = [_whole(kind, raw)[0] for kind in (StaggeredSolver, CoreSolver)]
    assert stopped[0].stop_reason == stopped[1].stop_reason == "velocity_step_below_tol"
    assert len(stopped[1].residual_history) == len(stopped[0].residual_history) < 40


# ---- The core's own solves ------------------------------------------------------


def _solve_raw(name: str, mode: str = "coupled", **solver: object) -> dict:
    raw = _raw(name, mode, **solver)
    return raw


def _whole(kind: type, raw: dict, **keys: object) -> tuple[StaggeredSolver, list]:
    """A solve from rest to its stop, its checkpoints kept."""
    mesh, config = _configured(copy.deepcopy(raw))
    solver = _solver(kind, mesh, config)
    kept: list[SteadyCheckpoint] = []
    with np.errstate(all="ignore"):
        solver.solve_steady(on_checkpoint=kept.append, **keys)
    return solver, kept


def _assert_same_solve(a: StaggeredSolver, b: StaggeredSolver) -> None:
    assert a.stop_reason == b.stop_reason
    assert _same(a.residual_history, b.residual_history)
    assert a.pressure_cap_hits == b.pressure_cap_hits
    assert a.flow_state is not None and b.flow_state is not None
    for field in FIELDS:
        x, y = getattr(a.flow_state, field), getattr(b.flow_state, field)
        assert (x is None) == (y is None)
        if x is not None:
            assert _same(x, y), field
    assert a.flow_state.outer == b.flow_state.outer


CASES = [(ROOM, "coupled"), ("cavity", "laminar")]


@pytest.mark.parametrize(("name", "mode"), CASES)
def test_a_core_solve_resumed_from_its_checkpoint_is_the_uninterrupted_one(
    name: str, mode: str
) -> None:
    """Interrupted at a checkpoint and resumed by the core: the same solve, bit for bit.

    Resumed from the second checkpoint and from the last before the stop. A solve
    resumed more than a window before its stop refills its window with the same
    steps by then, so a resume that dropped the window would stop where the
    uninterrupted solve stops; from the last checkpoint the stop falls inside the
    first window, which the checkpoint's own steps fill (prompt 95's plants).
    """
    raw = _solve_raw(name, mode, checkpoint_interval=150)
    whole, kept = _whole(CoreSolver, raw)
    assert whole.converged and len(kept) >= 2
    assert [c.state.outer for c in kept] == [150 * (i + 1) for i in range(len(kept))]
    assert whole.flow_state.outer - kept[-1].state.outer < RATE_WINDOW
    for checkpoint in (kept[1], kept[-1]):
        resumed, _ = _whole(CoreSolver, raw, resume=checkpoint)
        _assert_same_solve(resumed, whole)
        assert (
            resumed.steady_checkpoint.velocity_steps
            == whole.steady_checkpoint.velocity_steps
        )


@pytest.mark.parametrize(("name", "mode"), CASES)
def test_a_solve_with_on_iteration_is_the_solve_without_it(
    name: str, mode: str
) -> None:
    """One outer iteration per call while a callback reads them: the same bits."""
    raw = _solve_raw(name, mode, max_simple_iter=260)
    plain, _ = _whole(CoreSolver, raw)
    mesh, config = _configured(copy.deepcopy(raw))
    watched = _solver(CoreSolver, mesh, config)
    seen = []
    watched.solve_steady(on_iteration=seen.append)
    _assert_same_solve(watched, plain)
    assert plain.core_calls == 1 and watched.core_calls == 260
    assert [s.iteration for s in seen] == list(range(260))
    assert _same([s.residual for s in seen], plain.residual_history)
    last = to_cell_centers(plain.flow_state.u, plain.flow_state.v)
    assert _same(seen[-1].u, last[0]) and _same(seen[-1].p, plain.flow_state.p)


def test_a_laminar_run_planted_to_diverge_ends_at_its_cap_as_the_reference() -> None:
    """Non-finite fields run to the cap, stop as the reference stops and keep no state."""
    mesh, config = _configured(
        _raw("cavity", "laminar", max_simple_iter=6, max_pressure_iter=40)
    )
    u = np.zeros((CAVITY, CAVITY + 1))
    u[CAVITY // 2, CAVITY // 2] = 1e200
    initial = FlowState(
        u=u, v=np.zeros((CAVITY + 1, CAVITY)), p=np.zeros((CAVITY, CAVITY))
    )
    solved = []
    for kind in (StaggeredSolver, CoreSolver):
        solver = _solver(kind, mesh, config)
        with np.errstate(all="ignore"):
            solver.solve_steady(initial=initial)
        solved.append(solver)
    ref, core = solved
    assert core.stop_reason == ref.stop_reason == CAP_STOP
    assert core.flow_state is None and core.steady_checkpoint is None
    assert ref.flow_state is None
    assert len(core.residual_history) == len(ref.residual_history) == 6
    assert np.isnan(core.residual_history[-1]) and np.isnan(ref.residual_history[-1])
    assert core.pressure_cap_hits == ref.pressure_cap_hits


@pytest.mark.usefixtures("libm_numpy")
def test_a_positivity_failure_in_the_coupled_solve_raises_as_the_reference() -> None:
    """A Courant number far above one: the step's k goes negative at an outer iteration.

    The reference's message names the outer iteration and the first bad cell; with
    the core's dots the two sides' fields are the same bits, so the messages are.
    """
    raws = _raw(ROOM, "coupled", pressure_preconditioner="jacobi", max_pressure_iter=2)
    raws["turbulence"]["cfl_number"] = 0.5
    mesh, config = _configured(raws)
    raised = []
    histories = []
    for kind in (StaggeredSolver, CoreSolver):
        solver = _solver(kind, mesh, config)
        object.__setattr__(config.turbulence, "cfl_number", 40.0)
        if solver._model is not None:
            solver._model._cfl = 40.0
        if isinstance(solver, CoreSolver):
            solver._modules = core_mapping.flow_modules(
                mesh, config, solver._boundary, solver._corrector
            )
        with (
            _core_dots(),
            np.errstate(all="ignore"),
            pytest.raises(PositivityError) as err,
        ):
            solver.solve_steady()
        raised.append(err.value)
        histories.append((list(solver.residual_history), solver.pressure_cap_hits))
    assert str(raised[1]) == str(raised[0])
    assert str(raised[0]).startswith("outer iteration ")
    assert _same(raised[1].minimum, raised[0].minimum)
    # Every correction reaches its cap of two, the failing iteration's among them,
    # which the reference counts before it raises.
    assert histories[1] == histories[0]
    assert histories[0][1] == len(histories[0][0]) + 1


def test_formed_conditions_the_step_refuses_end_the_solve_as_the_reference() -> None:
    """A discharge vent of no turbulence intensity carries in k and eps of zero.

    ``SimConfig`` allows no intensity of zero, so the segments are changed after it
    has validated them (prompt 94's room). The supply's own is positive, so the
    solve starts; its first k-epsilon step refuses the conditions it is handed, on
    the vent faces, and so does the core's outer iteration. The correction is
    capped at two, so that the failing iteration's own correction, which the
    reference has counted before it raises, is a cap hit on both sides.
    """
    raw = _raw("80x30-vents-shelf", "coupled", max_simple_iter=3, max_pressure_iter=2)
    mesh, config = _configured(raw)
    vents = [
        run.name for run in StaggeredBoundary(mesh, config).vent_runs() if run.is_inlet
    ]
    assert vents
    for segment in vents:
        object.__setattr__(config.boundaries[segment], "turbulence_intensity", 0.0)
    messages = []
    for kind in (StaggeredSolver, CoreSolver):
        solver = _solver(kind, mesh, config)
        with np.errstate(all="ignore"), pytest.raises(ValueError) as refused:
            solver.solve_steady()
        messages.append(
            (
                str(refused.value),
                len(solver.residual_history),
                solver.pressure_cap_hits,
            )
        )
    assert messages[1] == messages[0]
    assert "vent faces" in messages[0][0]
    assert messages[0][2] == 1


def _log_lines(records: list[logging.LogRecord]) -> list[tuple[str, str]]:
    """The solver's and the turbulence model's lines, the residual of each cut off."""
    return [
        (r.levelname, r.getMessage().split(" = ")[0])
        for r in records
        if r.name in ("src.solver_staggered", "src.turbulence")
    ]


@pytest.mark.parametrize(("name", "mode"), CASES)
def test_the_checkpoints_and_the_log_lines_fall_where_the_references_fall(
    name: str, mode: str, caplog: pytest.LogCaptureFixture
) -> None:
    """Checkpoints at the reference's outer iterations, and its log lines in its order."""
    raw = _solve_raw(name, mode, checkpoint_interval=50, max_simple_iter=230)
    lines = []
    kept = []
    for kind in (StaggeredSolver, CoreSolver):
        caplog.clear()
        with caplog.at_level(logging.INFO, logger="src.solver_staggered"):
            solver, checkpoints = _whole(kind, raw)
        lines.append(_log_lines(caplog.records))
        kept.append(
            [
                (c.state.outer, len(c.velocity_steps), c.rule_version)
                for c in checkpoints
            ]
        )
        assert solver.stop_reason == CAP_STOP
    assert (
        kept[1]
        == kept[0]
        == [(n, min(n, RATE_WINDOW), kept[0][0][2]) for n in (50, 100, 150, 200)]
    )
    assert lines[1] == lines[0]


# ---- Across the two sides --------------------------------------------------------


def _assert_level_two(got: StaggeredSolver, expected: StaggeredSolver) -> None:
    """ADR-017 H: the stop's reason, the outer count within one, each field inside its limit."""
    assert got.stop_reason == expected.stop_reason
    assert got.flow_state is not None and expected.flow_state is not None
    assert abs(got.flow_state.outer - expected.flow_state.outer) <= 1
    for field in FIELDS:
        x = getattr(expected.flow_state, field)
        if x is None:
            continue
        y = getattr(got.flow_state, field)
        assert float(np.max(np.abs(y - x))) <= FLOW_LIMIT * float(np.max(np.abs(x))), (
            field
        )


@pytest.mark.parametrize(("name", "mode"), CASES)
@pytest.mark.parametrize("taken_by", ["reference", "core"])
def test_a_checkpoint_of_either_side_resumed_by_the_other_reaches_its_stop(
    name: str, mode: str, taken_by: str
) -> None:
    """A reference checkpoint resumed by the core, and a core one by the reference: level 2.

    From the last checkpoint before the stop, so that the stop reads the window the
    checkpoint carried across from the other side.
    """
    raw = _solve_raw(name, mode, checkpoint_interval=150)
    sides = {"reference": StaggeredSolver, "core": CoreSolver}
    other = "core" if taken_by == "reference" else "reference"
    whole, kept = _whole(sides[taken_by], raw)
    reference, _ = _whole(StaggeredSolver, raw) if taken_by == "core" else (whole, None)
    last = kept[-1]
    assert reference.flow_state.outer - last.state.outer < RATE_WINDOW
    resumed, _ = _whole(sides[other], raw, resume=last)
    _assert_level_two(resumed, reference)
    assert resumed.flow_state.outer > last.state.outer


def test_a_warm_start_is_the_references_at_level_two() -> None:
    """A converged state warm-started under its own boundaries: both stop when the window fills."""
    raw = _solve_raw(ROOM, "coupled")
    reference, _ = _whole(StaggeredSolver, raw)
    state = reference.flow_state
    assert state is not None
    warm = [
        _whole(kind, raw, initial=state)[0] for kind in (StaggeredSolver, CoreSolver)
    ]
    _assert_level_two(warm[1], warm[0])
    assert warm[1].flow_state.outer == warm[0].flow_state.outer == RATE_WINDOW


# ---- The binding's refusals ------------------------------------------------------


def _flow_parts(mode: str = "laminar") -> tuple:
    mesh, config = _configured(_raw("cavity", mode, max_simple_iter=5))
    solver = _solver(StaggeredSolver, mesh, config)
    modules = core_mapping.flow_modules(
        mesh, config, solver._boundary, solver._corrector
    )
    return mesh, config, solver, modules


def test_flow2_refuses_modules_of_another_grid_or_mode() -> None:
    """Outlets of another grid, a model without its boundary, a rule of no name."""
    mesh, config, solver, modules = _flow_parts()
    other = core_mapping.outlet_faces(mesh, solver._boundary)
    arguments = [
        modules.grid,
        modules.predictor,
        modules.corrector,
        modules.outlets,
        None,
        None,
        FLUID,
        config.mu,
        config.rho,
        config.mu / config.rho,
        1.0,
        solver.reference_velocity,
        ("velocity_step", 1e-6),
        None,
    ]
    _cfdcore.Flow2(*arguments)
    for at, value, error, message in [
        (3, other, ValueError, "another grid"),
        (12, ("stop", 1e-6), ValueError, "'velocity_step' or 'error_estimate'"),
        (12, ("velocity_step",), TypeError, "convergence_tol"),
        (12, ("velocity_step", -1.0), ValueError, "bound"),
        (7, 0.0, ValueError, "not finite and positive"),
        (6, 99, ValueError, "no cell is FLUID"),
    ]:
        changed = list(arguments)
        changed[at] = value
        with pytest.raises(error, match=message):
            _cfdcore.Flow2(*changed)


def test_flow2s_state_and_chunks_are_refused_when_they_do_not_fit() -> None:
    """Fields of another shape, k for a laminar solve, a chunk of no iteration."""
    _, config, solver, modules = _flow_parts()
    core = core_mapping.flow_solver(
        modules, config, solver._boundary, solver.reference_velocity, solver.flux_scale
    )
    n = CAVITY
    u, v, p = np.zeros((n, n + 1)), np.zeros((n + 1, n)), np.zeros((n, n))
    core.flow_state(u, v, p)
    with pytest.raises(TypeError, match="is not"):
        core.flow_state(np.zeros((n, n)), v, p)
    with pytest.raises(ValueError, match="exactly when the solve is coupled"):
        core.flow_state(u, v, p, p, p, p)
    with pytest.raises(ValueError, match="at least one outer iteration"):
        core.solve_steady(0, 0)
    with pytest.raises(ValueError, match="not one the rule restores"):
        core.restore((1e-3,), (1e-6,))
    chunk = core.solve_steady(2, 0)
    assert chunk["outer"] == 2 and chunk["failure"] is None
    assert core.rule_version == 3
    fields = core.flow_fields()
    assert set(fields) == {"u", "v", "p"}


def test_a_chunks_failure_maps_to_the_references_exception() -> None:
    """Each failure the core returns raises the type and the words the reference raises."""
    assert isinstance(
        core_mapping.flow_error(
            {"type": "ValueError", "message": "faces must be finite"}
        ),
        ValueError,
    )
    error = core_mapping.flow_error(
        {
            "type": "PositivityError",
            "iteration": 7,
            "quantity": "k",
            "count": 2,
            "row": 1,
            "column": 3,
            "value": -1.5,
            "minimum": -1.5,
        }
    )
    assert isinstance(error, PositivityError)
    assert str(error).startswith("outer iteration 7: k is not positive and finite at 2")
    assert error.minimum == -1.5
    assert isinstance(
        core_mapping.flow_error({"type": "ZeroDivisionError", "message": ""}),
        ZeroDivisionError,
    )
    assert isinstance(
        core_mapping.flow_error({"type": "LinAlgError", "message": ""}),
        np.linalg.LinAlgError,
    )


# ---- Where the stop binds --------------------------------------------------------
#
# Each solve above stops where the estimates (a) and (e) first hold and continuity
# holds with them, so a stop that read only (a) and (e), or one that read the wrong
# imbalance, would stop where the reference does. These solves make continuity the
# condition that binds: the estimates hold a hundred outer iterations before the
# stop, and the solve either stops later on (b), (c) and (d) or runs to its cap.


def _whole_pair(
    raw: dict, caplog: pytest.LogCaptureFixture, **keys: object
) -> tuple[
    list[StaggeredSolver], list[list[tuple[str, str]]], list[list[SteadyCheckpoint]]
]:
    """A solve to its stop on each side from the same keys, given the core's dot product.

    The raw configuration names the diagonal preconditioner, so both chains are the
    same bits and a stop is compared exactly. Returns the solvers (the reference's
    first), their log lines and their checkpoints.
    """
    solvers, lines, kept = [], [], []
    for kind in (StaggeredSolver, CoreSolver):
        caplog.clear()
        with (
            caplog.at_level(logging.INFO, logger="src.solver_staggered"),
            _core_dots(),
        ):
            solver, checkpoints = _whole(kind, raw, **keys)
        solvers.append(solver)
        lines.append(_log_lines(caplog.records))
        kept.append(checkpoints)
    return solvers, lines, kept


def _estimates_first_hold(solver: CoreSolver, bound: float) -> int | None:
    """The outer iteration, counted from one, where (a) and with it on (e) first hold."""
    readings = solver.rule_readings
    for n, estimate in enumerate(readings["estimate"]):
        viscosity = readings["viscosity_estimate"][n]
        if estimate < bound and (math.isnan(viscosity) or viscosity < bound):
            return n + 1
    return None


@pytest.mark.parametrize(
    ("name", "mode", "cap", "tolerance", "stop"),
    [
        pytest.param(
            "cavity",
            "laminar",
            2,
            1e-5,
            "error_estimate_and_continuity",
            id="cavity-later",
        ),
        pytest.param("cavity", "laminar", 2, 1e-12, CAP_STOP, id="cavity-to-the-cap"),
        pytest.param(
            ROOM, "coupled", 4, 1e-5, "error_estimate_and_continuity", id="room-later"
        ),
        pytest.param(ROOM, "coupled", 4, 1e-8, CAP_STOP, id="room-to-the-cap"),
    ],
)
def test_the_stop_waits_for_continuity_as_the_reference_waits(
    request: pytest.FixtureRequest,
    caplog: pytest.LogCaptureFixture,
    name: str,
    mode: str,
    cap: int,
    tolerance: float,
    stop: str,
) -> None:
    """(a) and (e) hold long before the stop, and (b), (c) and (d) put it off.

    The correction is capped at a few conjugate-gradient iterations and
    ``iteration_error_tol`` is 1, so that the estimates hold about a hundred outer
    iterations in while the imbalance is still above ``mass_imbalance_tol``. With the
    tolerance at 1e-5 it falls below it later and the solve stops there, on both
    sides at the same outer iteration; at 1e-8 or 1e-12 it never does and both run to
    their cap. A stop that dropped one of the conditions, or read the imbalance of
    another iterate, would stop where the estimates first hold. The log lines are
    compared too, the last two at a stop among them.
    """
    if mode == "coupled":
        request.getfixturevalue("libm_numpy")
    raw = _raw(
        name,
        mode,
        pressure_preconditioner="jacobi",
        max_pressure_iter=cap,
        max_simple_iter=260,
        iteration_error_tol=1.0,
        mass_imbalance_tol=tolerance,
    )
    (ref, core), lines, _ = _whole_pair(raw, caplog)
    assert ref.stop_reason == core.stop_reason == stop
    _assert_same_solve(ref, core)
    held = _estimates_first_hold(core, 1.0)
    assert held is not None
    assert held + 20 <= len(core.residual_history)
    assert lines[1] == lines[0]
    at = len(core.residual_history) - 1
    if stop == CAP_STOP:
        assert lines[1][-1] == ("WARNING", "Not converged: stopped at max_simple_iter")
    else:
        assert lines[1][-2:] == [
            ("INFO", f"SIMPLE iter {at:4d}: residual"),
            ("INFO", f"Converged at iteration {at}"),
        ]


def test_a_velocity_step_stops_between_its_bound_and_twice_it(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The first step below the bound is later than the first below twice the bound.

    The cavity's own residuals, from a solve that cannot stop, give the two: the
    bound is 0.9 of the residual at outer iteration 60, which is the lowest so far,
    so the stop under the bound comes after 60 and the stop under twice the bound
    at or before it. A bound doubled in the core's set-up stops at the second.
    """
    keys = {
        "pressure_preconditioner": "jacobi",
        "stopping_rule": "velocity_step",
        "max_simple_iter": 200,
    }
    probe = _raw("cavity", "laminar", convergence_tol=1e-300, **keys)
    with _core_dots():
        history = _whole(StaggeredSolver, probe)[0].residual_history
    at = 60
    assert history[at] <= min(history[:at])
    bound = 0.9 * history[at]
    first_below = next(n for n, r in enumerate(history) if r < bound)
    first_below_twice = next(n for n, r in enumerate(history) if r < 2.0 * bound)
    assert first_below_twice <= at < first_below
    raw = _raw("cavity", "laminar", convergence_tol=bound, **keys)
    (ref, core), lines, _ = _whole_pair(raw, caplog)
    assert ref.stop_reason == core.stop_reason == "velocity_step_below_tol"
    assert ref.pressure_cap_hits == 0
    assert len(core.residual_history) == len(ref.residual_history) == first_below + 1
    _assert_same_solve(ref, core)
    assert lines[1] == lines[0]


# ---- The outlets inside the outer iteration --------------------------------------
#
# No room above has a pressure outlet but the small room's, in the cases that name
# it: the stage that copies each outlet face's interior neighbour before a
# prediction is read by nothing else, so a solve that left it out, or one edge of
# it, would pass every other case.


def test_the_small_room_is_open_on_every_edge() -> None:
    """The small room has a pressure outlet on each of its four edges."""
    mesh, config = _configured(_raw(SMALL, "coupled"))
    outlets = StaggeredBoundary(mesh, config).pressure_outlets()
    assert {
        edge: bool(outlets[edge].is_outlet.any()) for edge in core_mapping.WALL_EDGES
    } == {edge: True for edge in core_mapping.WALL_EDGES}


def test_the_poiseuille_flow_solved_whole_is_the_references_at_level_two() -> None:
    """The channel's pressure outlet, read in every outer iteration of a whole solve.

    VAL-001's room at 40 by 20 cells: an inlet on the left, a pressure outlet on the
    right, and walls. Both sides stop for the same reason within one outer
    iteration, with every field inside level 2's limit (ADR-017 H).
    """
    with open("configs/validation_poiseuille.yaml", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    raw["domain"].update(nx=40, ny=20)
    mesh, config = _configured(copy.deepcopy(raw))
    outlets = StaggeredBoundary(mesh, config).pressure_outlets()
    assert outlets["right"].is_outlet.all()
    ref, _ = _whole(StaggeredSolver, raw)
    core, _ = _whole(CoreSolver, raw)
    assert ref.converged and core.converged
    _assert_level_two(core, ref)


# ---- The solve's failure paths ---------------------------------------------------
#
# Each failure a planted state reaches through the core is compared with the
# reference's: the exception's type and message, the records the solve leaves, what
# the callbacks were handed before it raised, the log lines. Both sides take the
# core's dot product and the diagonal preconditioner, so the chains are the same
# bits and the iteration a failure falls at is the same.
#
# Failure kinds of ``flow.hpp`` that no state reaches, and why:
#
# - ``eddy_viscosity_negative``: the iterate is the step's ``C_mu k^2 / eps`` under
#   the relaxation, a mean of two values that are not negative; a warm start
#   rebuilds it from k and eps, and a resume refuses a negative one.
# - ``mu_eff_not_positive``: ``mu + rho nu_t`` is at least ``mu``, which is positive.
# - ``wall_mu_not_positive``: the wall function is a product of positive numbers over
#   a logarithm of at least ``ln(E y*_0)``, above one, and the smallest positive k
#   leaves its square root positive.
# - ``nothing_moves``: the faces after a correction are all zero only where no
#   boundary prescribes a velocity, and the rule that the turbulent configuration
#   needs refuses a velocity scale of zero when the solver is built.
# - ``pressure``: ``not_dominant`` and ``division_by_zero`` need a system built by
#   hand (tests/test_core_pressure.py); from a planted state they come only under the
#   multigrid preconditioner and an extreme field (u of 1e30), where the reference's
#   dot product and the core's part and the two sides do not fail together.
#   ``not_converged`` needs a coefficient that is NaN, whose answer NumPy gives
#   differently from one version to the next.


def _solver_at(
    kind: type, mesh: Mesh, config: SimConfig, cfl: float | None
) -> StaggeredSolver:
    """A solver whose Courant number is CFL, above what the configuration allows."""
    solver = _solver(kind, mesh, config)
    if cfl is None:
        return solver
    object.__setattr__(config.turbulence, "cfl_number", cfl)
    assert solver._model is not None
    solver._model._cfl = cfl
    if isinstance(solver, CoreSolver):
        solver._modules = core_mapping.flow_modules(
            mesh, config, solver._boundary, solver._corrector
        )
    return solver


def _warm(
    **spikes: tuple[tuple[int, int], float],
) -> Callable[[Mesh], dict[str, object]]:
    """A warm start from the room's planted state with each field's cell set to a value."""

    def start(_mesh: Mesh) -> dict[str, object]:
        initial = _planted(ROOM, "coupled", _rng("failures", "warm"))
        fields = {name: np.array(getattr(initial, name)) for name in FIELDS}
        for name, (cell, value) in spikes.items():
            fields[name][cell] = value
        return {"initial": FlowState(**fields)}

    return start


@functools.cache
def _checkpoint() -> SteadyCheckpoint:
    """A checkpoint of four outer iterations of the coupled room, the reference's."""
    mesh, config = _configured(
        _raw(ROOM, "coupled", pressure_preconditioner="jacobi", max_simple_iter=4)
    )
    solver = _solver(StaggeredSolver, mesh, config)
    with _core_dots(), np.errstate(all="ignore"):
        solver.solve_steady()
    assert solver.steady_checkpoint is not None
    return solver.steady_checkpoint


def _resumed(
    **spikes: tuple[list[tuple[int, int]], float],
) -> Callable[[Mesh], dict[str, object]]:
    """A resume of that checkpoint with each field's cells set to a value."""

    def start(_mesh: Mesh) -> dict[str, object]:
        base = _checkpoint()
        fields = {
            name: np.array(getattr(base.state, name)) for name in ("k", "eps", "nu_t")
        }
        for name, (cells, value) in spikes.items():
            for cell in cells:
                fields[name][cell] = value
        state = dataclasses.replace(base.state, stop_reason=None, **fields)
        return {"resume": dataclasses.replace(base, state=state)}

    return start


def _eps_spiked(seed: int) -> Callable[[Mesh], dict[str, object]]:
    """A warm start with eps set to a tiny value at one to three cells drawn from SEED.

    The step's eps goes negative in the first outer iteration, so the failure does
    not wait on a growth that another BLAS or NumPy could take elsewhere.
    """

    def start(_mesh: Mesh) -> dict[str, object]:
        rng = _rng("pos", seed, "eps_spike")
        initial = _planted(ROOM, "coupled", rng)
        fields = {name: np.array(getattr(initial, name)) for name in FIELDS}
        # The draws of a first cell are the reference's: its row and column are
        # drawn after the cell's value, as the right-hand side is evaluated first.
        rng.integers(1, 14), rng.integers(1, 39)
        for _ in range(int(rng.integers(1, 4))):
            value = 10.0 ** rng.uniform(-300.0, -5.0)
            fields["eps"][int(rng.integers(1, 14)), int(rng.integers(1, 39))] = value
        return {"initial": FlowState(**fields)}

    return start


def _fixed_overflow(mesh: Mesh) -> dict[str, object]:
    """A fixed eddy viscosity whose ``mu + rho nu_t`` overflows in one cell."""
    eddy = _eddy_viscosity(mesh, _rng("failures", "fixed"))
    eddy[3, 4] = 1.7e308
    return {"eddy_viscosity": eddy}


def _after_a_checkpoint_off_the_diagonal(found: re.Match[str], _ref: dict) -> None:
    """The failure falls after the first checkpoint, at a cell whose row and column differ."""
    assert int(found[1]) >= 4
    assert found[2] != found[3]


def _off_the_diagonal(found: re.Match[str], _ref: dict) -> None:
    """The failing cell's row and column differ."""
    assert found[2] != found[3]


def _records_the_failing_iteration(found: re.Match[str], ref: dict) -> None:
    """The reference has appended the failing iteration's residual and called back."""
    assert len(ref["history"]) == 1
    if ref["called_back"]:
        assert [event[0] for event in ref["events"]].count("iteration") == 1


POSITIVITY = (
    r"^outer iteration (\d+): {} is not positive and finite at \d+ non-SOLID cell\(s\), "
    r"first at row (\d+), column (\d+)"
)
PLANTED = [
    pytest.param(
        "coupled",
        {},
        _warm(eps=((7, 11), 1e-300)),
        None,
        ValueError,
        r"^nu_scale must be positive and finite, got inf$",
        _records_the_failing_iteration,
        id="nu-scale",
    ),
    pytest.param(
        "coupled",
        {},
        _warm(u=((5, 7), 1e80)),
        None,
        ValueError,
        r"^eddy_viscosity must be finite in every non-SOLID cell$",
        None,
        id="eddy-viscosity-not-finite",
    ),
    pytest.param(
        "coupled",
        {},
        _warm(u=((5, 7), 1e50)),
        None,
        ValueError,
        r"^faces must be finite$",
        None,
        id="faces-not-finite",
    ),
    pytest.param(
        "coupled",
        {"max_simple_iter": 8},
        _resumed(nu_t=([(7, 11)], 1.7e308)),
        None,
        ValueError,
        r"^mu_eff must be finite in every non-SOLID cell$",
        None,
        id="effective-viscosity-overflows",
    ),
    pytest.param(
        "coupled",
        {"max_simple_iter": 8},
        _resumed(k=([(0, 20), (0, 21)], 1.7e308)),
        None,
        ValueError,
        r"^wall_mu\['u'\] must be finite at every wall face$",
        None,
        id="wall-viscosity-of-u-not-finite",
    ),
    pytest.param(
        "coupled",
        {"max_simple_iter": 8},
        _resumed(k=([(5, 0), (6, 0)], 1.7e308)),
        None,
        ValueError,
        r"^wall_mu\['v'\] must be finite at every wall face$",
        None,
        id="wall-viscosity-of-v-not-finite",
    ),
    pytest.param(
        "coupled",
        {"max_simple_iter": 8},
        _resumed(k=([(7, 11)], 1e-100)),
        None,
        PositivityError,
        POSITIVITY.format("eps"),
        _after_a_checkpoint_off_the_diagonal,
        id="eps-after-a-resume-off-the-diagonal",
    ),
    pytest.param(
        "coupled",
        {"max_simple_iter": 14},
        _eps_spiked(14),
        5.0,
        PositivityError,
        POSITIVITY.format("eps"),
        _off_the_diagonal,
        id="eps-off-the-diagonal",
    ),
    pytest.param(
        "fixed",
        {},
        _fixed_overflow,
        None,
        ValueError,
        r"^mu_eff must be finite in every non-SOLID cell$",
        None,
        id="fixed-viscosity-overflows",
    ),
]


def _planted_outcome(
    kind: type,
    raw: dict,
    start: Callable[[Mesh], dict[str, object]],
    cfl: float | None,
    caplog: pytest.LogCaptureFixture,
    *,
    called_back: bool,
) -> dict:
    """What a solve from a planted state leaves: its error, records, events and log lines."""
    mesh, config = _configured(copy.deepcopy(raw))
    solver = _solver_at(kind, mesh, config, cfl)
    keys = start(mesh)
    events: list[tuple] = []

    def on_checkpoint(checkpoint: SteadyCheckpoint) -> None:
        events.append(
            (
                "checkpoint",
                checkpoint.state.outer,
                checkpoint.pressure_cap_hits,
                _bits(checkpoint.residual_history).tolist(),
                _bits(checkpoint.velocity_steps).tolist(),
            )
        )

    def on_iteration(state: IterationState) -> None:
        events.append(
            (
                "iteration",
                state.iteration,
                _bits(state.residual).item(),
                state.pressure_iterations,
                state.pressure_products,
                _bits(state.p).tobytes(),
            )
        )

    if called_back:
        keys["on_iteration"] = on_iteration
    caplog.clear()
    error = None
    with (
        caplog.at_level(logging.INFO, logger="src.solver_staggered"),
        _core_dots(),
        np.errstate(all="ignore"),
    ):
        try:
            solver.solve_steady(on_checkpoint=on_checkpoint, **keys)
        except Exception as raised:
            error = raised
    return {
        "error": None if error is None else (type(error).__name__, str(error)),
        "minimum": None
        if not hasattr(error, "minimum")
        else _bits(error.minimum).item(),
        "history": _bits(solver.residual_history).tolist(),
        "cap_hits": solver.pressure_cap_hits,
        "last_pressure_iterations": solver.last_pressure_iterations,
        "stop_reason": solver.stop_reason,
        "state": None if solver.flow_state is None else solver.flow_state.outer,
        "turbulence": solver.turbulence_state is None,
        "events": events,
        "called_back": called_back,
        "lines": _log_lines(caplog.records),
    }


@pytest.mark.parametrize(
    "called_back", [False, True], ids=["checkpoints", "on-iteration"]
)
@pytest.mark.parametrize(
    ("mode", "keys", "start", "cfl", "error", "message", "extra"), PLANTED
)
def test_a_failure_the_core_reaches_is_the_references(
    request: pytest.FixtureRequest,
    caplog: pytest.LogCaptureFixture,
    mode: str,
    keys: dict,
    start: Callable[[Mesh], dict[str, object]],
    cfl: float | None,
    error: type[Exception],
    message: str,
    extra: Callable[[re.Match[str], dict], None] | None,
    called_back: bool,
) -> None:
    """The same exception, records, callbacks and log lines, from a state that fails.

    Each case is run with checkpoints every three outer iterations, and again with
    ``on_iteration`` too, which cuts the core's chunks to one iteration. The failure
    kinds are the plan's: a refused scale of condition (e) (which has recorded its
    iteration and called back before it raises), an eddy viscosity that is not
    finite, faces that are not finite, an effective viscosity that overflows, a wall
    viscosity that is not finite on each component, k and eps not positive (one
    after a checkpoint, both off the diagonal) and a fixed viscosity that overflows.
    """
    if mode == "coupled":
        request.getfixturevalue("libm_numpy")
    raw = _raw(
        ROOM,
        mode,
        **{
            "pressure_preconditioner": "jacobi",
            "max_simple_iter": 6,
            "checkpoint_interval": 3,
            **keys,
        },
    )
    if cfl is not None:
        raw["turbulence"]["cfl_number"] = 0.5
    ref, core = (
        _planted_outcome(kind, raw, start, cfl, caplog, called_back=called_back)
        for kind in (StaggeredSolver, CoreSolver)
    )
    assert ref["error"] is not None
    assert ref["error"][0] == error.__name__
    found = re.search(message, ref["error"][1])
    assert found is not None, ref["error"]
    if extra is not None:
        extra(found, ref)
    assert core == ref


def test_the_core_drivers_set_up_refusals_are_the_references() -> None:
    """Eddy viscosity beside the turbulence section, and a warm start beside a resume."""
    mesh, config = _configured(_raw(ROOM, "coupled", max_simple_iter=2))
    eddy = _eddy_viscosity(mesh, _rng("refusals"))
    messages = []
    for kind in (StaggeredSolver, CoreSolver):
        with pytest.raises(ValueError) as refused:
            _solver(kind, mesh, config).solve_steady(eddy_viscosity=eddy)
        messages.append(str(refused.value))
    assert messages[1] == messages[0]
    assert "refused with the turbulence section present" in messages[0]
    mesh, config = _configured(
        _raw("cavity", "laminar", max_simple_iter=2, checkpoint_interval=1)
    )
    first = _solver(StaggeredSolver, mesh, config)
    kept: list[SteadyCheckpoint] = []
    first.solve_steady(on_checkpoint=kept.append)
    assert first.flow_state is not None and kept
    messages = []
    for kind in (StaggeredSolver, CoreSolver):
        with pytest.raises(ValueError) as refused:
            _solver(kind, mesh, config).solve_steady(
                initial=first.flow_state, resume=kept[0]
            )
        messages.append(str(refused.value))
    assert messages[1] == messages[0]
    assert "initial and resume are exclusive" in messages[0]


# ---- Callbacks, checkpoints, records and warnings --------------------------------


@pytest.mark.parametrize(("name", "mode"), CASES)
def test_each_iterations_state_and_each_checkpoint_is_the_references(
    request: pytest.FixtureRequest, name: str, mode: str
) -> None:
    """``on_iteration``'s fields and the checkpoints written beside it, bit for bit.

    Five outer iterations from a planted state with a checkpoint every two: the
    callback is handed the iteration, the residual, the correction's iterations and
    products and the cell-centred velocities and pressure; the checkpoints at 2 and
    4 carry the window, the history, the cap hits and the fields.
    """
    if mode == "coupled":
        request.getfixturevalue("libm_numpy")
    initial = _planted(name, mode, _rng("states", name, mode))
    mesh, config = _configured(
        _raw(
            name,
            mode,
            pressure_preconditioner="jacobi",
            max_simple_iter=5,
            checkpoint_interval=2,
        )
    )
    states: list[list[IterationState]] = [[], []]
    kept: list[list[SteadyCheckpoint]] = [[], []]
    for side, kind in enumerate((StaggeredSolver, CoreSolver)):
        with _core_dots(), np.errstate(all="ignore"):
            _solver(kind, mesh, config).solve_steady(
                states[side].append, initial=initial, on_checkpoint=kept[side].append
            )
    ref_states, core_states = states
    assert [s.iteration for s in core_states] == list(range(5))
    for a, b in zip(core_states, ref_states, strict=True):
        assert (a.iteration, a.pressure_iterations, a.pressure_products) == (
            b.iteration,
            b.pressure_iterations,
            b.pressure_products,
        )
        assert _same(a.residual, b.residual)
        for field in ("u", "v", "p"):
            assert _same(getattr(a, field), getattr(b, field)), field
    # The correction's products and its iterations are two counts, so that one given
    # for the other is seen.
    assert any(s.pressure_products != s.pressure_iterations for s in ref_states)
    assert (
        [c.state.outer for c in kept[1]] == [c.state.outer for c in kept[0]] == [2, 4]
    )
    for a, b in zip(kept[1], kept[0], strict=True):
        assert a.pressure_cap_hits == b.pressure_cap_hits
        assert a.velocity_steps == b.velocity_steps
        assert a.viscosity_steps == b.viscosity_steps
        assert _same(a.residual_history, b.residual_history)
        for field in FIELDS:
            x, y = getattr(a.state, field), getattr(b.state, field)
            assert (x is None) == (y is None)
            if x is not None:
                assert _same(x, y), field


def test_a_resumed_solve_carries_its_cap_hits_on(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Every correction reaches its cap of two, so the hits a resume carries are not zero.

    The three checkpoints hold three, six and nine hits; the solve resumed from each
    reaches the cap on all eleven of its corrections, counted from the checkpoint's.
    """
    raw = _raw(
        "cavity",
        "laminar",
        pressure_preconditioner="jacobi",
        max_pressure_iter=2,
        max_simple_iter=11,
        checkpoint_interval=3,
    )
    with _core_dots():
        first, kept = _whole(StaggeredSolver, raw)
    assert first.pressure_cap_hits == 11
    assert [c.pressure_cap_hits for c in kept] == [3, 6, 9]
    for checkpoint in kept:
        (ref, core), lines, _ = _whole_pair(raw, caplog, resume=checkpoint)
        assert ref.pressure_cap_hits == core.pressure_cap_hits == 11
        _assert_same_solve(ref, core)
        assert lines[1] == lines[0]


@pytest.mark.parametrize("mode", MODES)
def test_the_solver_leaves_the_references_faces_and_turbulence_state(
    request: pytest.FixtureRequest, mode: str
) -> None:
    """``face_velocities`` and ``turbulence_state`` after a solve, for the transport and the march."""
    if mode == "coupled":
        request.getfixturevalue("libm_numpy")
    ref, core = _pair_of_solves(ROOM, mode, 3, dots=True)
    assert ref.face_velocities is not None and core.face_velocities is not None
    assert _same(core.face_velocities.u, ref.face_velocities.u)
    assert _same(core.face_velocities.v, ref.face_velocities.v)
    assert (core.turbulence_state is None) == (ref.turbulence_state is None)
    assert (ref.turbulence_state is None) == (mode != "coupled")
    if ref.turbulence_state is not None:
        for field in ("k", "eps", "nu_t"):
            assert _same(
                getattr(core.turbulence_state, field),
                getattr(ref.turbulence_state, field),
            ), field


@pytest.mark.usefixtures("libm_numpy")
@pytest.mark.parametrize(
    ("max_iter", "spiked"),
    [
        pytest.param(1, False, id="a-cap-of-one"),
        pytest.param(3, False, id="a-cap-of-three"),
        pytest.param(2, True, id="a-step-that-then-fails-its-signs"),
    ],
)
def test_the_k_epsilon_steps_warnings_are_the_references(
    caplog: pytest.LogCaptureFixture, max_iter: int, spiked: bool
) -> None:
    """The two warnings of ``KEpsilonModel.step``, from what each chunk returns.

    A conjugate-gradient cap of one to three leaves both solves capped, so each step
    falls back to Jacobi and stops at the cap, and the reference logs both warnings
    every iteration, before the iteration's own lines. The last case resumes a
    state whose k is tiny in one cell, so that the step then fails its signs at
    the first iteration: the warnings were logged before it raised, and the
    core's must have been too.
    """
    raw = _raw(ROOM, "coupled", pressure_preconditioner="jacobi", max_simple_iter=8)
    raw["turbulence"]["max_iter"] = max_iter
    mesh, config = _configured(raw)
    if spiked:
        keys = _resumed(k=([(7, 11)], 1e-100))(mesh)
    else:
        keys = {"initial": _planted(ROOM, "coupled", _rng("warnings", max_iter))}
    outcomes = []
    for kind in (StaggeredSolver, CoreSolver):
        solver = _solver(kind, mesh, config)
        caplog.clear()
        error = None
        with caplog.at_level(logging.INFO), _core_dots(), np.errstate(all="ignore"):
            try:
                solver.solve_steady(**keys)
            except PositivityError as raised:
                error = str(raised)
        outcomes.append((_log_lines(caplog.records), error))
    assert outcomes[1] == outcomes[0]
    lines = [text for _, text in outcomes[0][0]]
    fell_back = [t for t in lines if t.startswith("CG k or eps solve fell back")]
    stopped = [t for t in lines if t.startswith("implicit k or eps solve stopped")]
    assert fell_back and stopped
    assert all(f"max_iter {max_iter} " in t for t in stopped)
    assert (outcomes[0][1] is not None) == spiked
    if spiked:
        # The failing iteration's warnings come before the exception, after its cap's.
        assert lines[-1].startswith("implicit k or eps solve stopped")


def test_steady_solves_in_the_core_builds_the_core_solver_in_a_scenario_run(
    tmp_path: Path,
) -> None:
    """A scenario's steady solve, run inside the patch, is the core's and is kept.

    The product room at 80 by 30 cells, its first interval's flow state: solved by a
    ``CoreSolver`` (counted by its ``solve_steady``), stopped by the rule, and
    written to the run file. The patch is undone on leaving, also by an exception.
    Step 6 of ADR-017 replaces the patch with the ``solver.backend`` key; until then
    the probe and the parity tests share it, and this is its only test.
    """

    class StopError(Exception):
        """Ends the run once its flow states are solved."""

    def stop(_step: object) -> None:
        raise StopError

    source = Path("configs/scenario_equipment_dust.yaml")
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    raw["scenario"]["base"] = str((source.parent / raw["scenario"]["base"]).resolve())
    raw["domain"] = {"nx": 80, "ny": 30}
    cell = (8.0 / 80) * (3.0 / 30)
    raw.setdefault("solver", {})["mass_imbalance_tol"] = 1.0e-4 * 1.2 * cell / 60.0
    config = SimConfig.from_dict(raw, "configs")
    reference = time_integration.StaggeredSolver
    calls: list[CoreSolver] = []
    original = CoreSolver.solve_steady

    def counted(solver: CoreSolver, *args: object, **keys: object) -> object:
        calls.append(solver)
        return original(solver, *args, **keys)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(CoreSolver, "solve_steady", counted)
        with steady_solves_in_the_core(), pytest.raises(StopError):
            assert time_integration.StaggeredSolver is CoreSolver
            time_integration.run(config, tmp_path / "run.h5", on_step=stop)
    assert time_integration.StaggeredSolver is reference
    assert len(calls) == 1 and all(isinstance(s, CoreSolver) for s in calls)
    assert calls[0].converged
    assert calls[0].stop_reason == "error_estimate_and_continuity"
    run = RunFile(tmp_path / "run.h5")
    (fingerprint,) = run.flow_fingerprints()
    record = run.read_flow_state(fingerprint)
    assert record.state.stop_reason == calls[0].stop_reason
    assert record.state.outer == len(calls[0].residual_history)
    with (
        pytest.raises(RuntimeError, match="raised inside"),
        steady_solves_in_the_core(),
    ):
        raise RuntimeError("raised inside")
    assert time_integration.StaggeredSolver is reference


# ---- The binding's refusals, continued -------------------------------------------


def test_flow2_refuses_a_fixed_viscosity_as_the_predictor_refuses_it() -> None:
    """A fixed ``mu_eff`` that is not finite or not positive in a non-SOLID cell.

    The core reads it when the solver is built, and the mapping hands the refusal
    on in the words of ``predict``'s own (``_check_mu_eff``). A SOLID cell is not
    read, so a NaN there builds a solver.
    """
    mesh, config = _configured(_raw("80x30", "laminar", max_simple_iter=2))
    solver = _solver(StaggeredSolver, mesh, config)
    modules = core_mapping.flow_modules(
        mesh, config, solver._boundary, solver._corrector
    )
    good = solver._effective_viscosity(_eddy_viscosity(mesh, _rng("fixed viscosity")))

    def built(mu_eff: np.ndarray) -> object:
        return core_mapping.flow_solver(
            modules,
            config,
            solver._boundary,
            solver.reference_velocity,
            solver.flux_scale,
            mu_eff,
        )

    built(good)
    solid = np.argwhere(mesh.cell_type == SOLID)
    live = np.argwhere(mesh.cell_type != SOLID)
    assert len(solid) and len(live)
    spoilt = np.array(good)
    spoilt[tuple(solid[0])] = math.nan
    built(spoilt)
    u, v, p = allocate_fields(mesh)
    for bad in (math.nan, math.inf, -math.inf, 0.0, -1e-3):
        spoilt = np.array(good)
        spoilt[tuple(live[len(live) // 2])] = bad
        with pytest.raises(ValueError) as expected:
            solver._predictor.predict(u, v, p, mu_eff=spoilt)
        with pytest.raises(ValueError) as got:
            built(spoilt)
        assert str(got.value) == str(expected.value)


def test_the_rule_versions_are_handed_in_and_each_refused_below_one() -> None:
    """The core holds no copy of the reference's two rule versions.

    A rule given 7 and 11 reports 7 with (e) off and 11 with it on; a version
    below one, not an int or a bool is refused at the binding, naming it.
    """
    arguments = [RATE_WINDOW, 7, 11, 0.45, 1.3, 1e-6, 1e-9]
    assert _cfdcore.StoppingRule2(*arguments).version == 7
    assert _cfdcore.StoppingRule2(*arguments, True).version == 11
    for position, name in ((1, "version_without_e"), (2, "version_with_e")):
        for bad, error in (
            (0, ValueError),
            (-3, ValueError),
            (True, TypeError),
            (3.0, TypeError),
        ):
            spoilt = list(arguments)
            spoilt[position] = bad
            with pytest.raises(error, match=name):
                _cfdcore.StoppingRule2(*spoilt)
    _, config, solver, modules = _flow_parts()
    with pytest.raises(TypeError, match="version_without_e"):
        _cfdcore.Flow2(
            modules.grid,
            modules.predictor,
            modules.corrector,
            modules.outlets,
            None,
            None,
            FLUID,
            config.mu,
            config.rho,
            config.mu / config.rho,
            1.0,
            solver.reference_velocity,
            ("error_estimate", RATE_WINDOW, 0.45, 1.3, 1e-6, 1e-9),
            None,
        )


def test_a_window_restored_under_the_velocity_step_is_refused_as_no_rule() -> None:
    """The velocity step has no window: an empty one restores nothing, another is refused."""
    raw = _raw("cavity", "laminar", stopping_rule="velocity_step", max_simple_iter=5)
    mesh, config = _configured(raw)
    solver = _solver(StaggeredSolver, mesh, config)
    modules = core_mapping.flow_modules(
        mesh, config, solver._boundary, solver._corrector
    )
    core = core_mapping.flow_solver(
        modules, config, solver._boundary, solver.reference_velocity, None
    )
    assert core.rule_version is None and core.rule_keys is None
    core.restore((), ())
    for steps, viscosity in (((1e-3,), ()), ((), (1e-6,))):
        with pytest.raises(ValueError, match="restored under velocity_step"):
            core.restore(steps, viscosity)
