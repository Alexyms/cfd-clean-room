"""The compiled core's stopping rule and outer iteration against the NumPy reference.

REQ-N03, ADR-017 D, G and H; prompt 95. ``_cfdcore.StoppingRule2`` against
``ErrorEstimateRule`` on synthetic histories: every verdict, window and scale bit
for bit, and each estimate to 1e-12 of the reference's, since its rate fit's
``x @ log(steps)`` is OpenBLAS's dot there and the core's fixed blocks here (the
second kind). ``viscosity_scale``'s median, the imbalance summary, the cell
centres and the max-norms bit for bit. Then ``tests/core_driver.CoreSolver``,
whose outer iterations run in a ``_cfdcore.Flow2``, against ``StaggeredSolver``:
outer iterations from planted fields on prompt 87's meshes and the cavity's,
laminar, under a fixed eddy viscosity and coupled, bit for bit with the
reference's conjugate gradients given the core's dot product (the chain judged
block by block, each block handed what the other side's was), and within level
one's 1e-12 as the reference runs; the core's own restarts, callbacks, divergence
and positivity failure; and checkpoints that cross between the sides, judged at
level 2 (ADR-017 H).

The coupled blocks call the C library's logarithm and powers, so the tests that
compare them bit for bit ask for ``libm_numpy`` (prompt 94). Skipped with its
reason when ``_cfdcore`` cannot be imported, an error instead with
``CFD_CORE_REQUIRED=1``, as ``tests/test_core_parity.py``.
"""

import contextlib
import copy
import logging
import math
import os
from collections.abc import Iterator
from fractions import Fraction

import numpy as np
import pytest
import yaml

from src import pressure, scalar_scheme
from src.boundary_staggered import StaggeredBoundary
from src.config import SimConfig
from src.mesh import FLUID, SOLID, Mesh
from src.solver_staggered import CAP_STOP, FlowState, StaggeredSolver, SteadyCheckpoint
from src.staggered import to_cell_centers
from src.stopping import (
    RATE_WINDOW,
    ErrorEstimateRule,
    ImbalanceSummary,
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
from tests.core_driver import CoreSolver
from tests.test_core_parity import MESHES, _raw_config
from tests.test_core_turbulence import _raw as _turbulence_raw
from tests.test_core_turbulence import (
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
    raw = _cavity_raw(mode) if name == "cavity" else copy.deepcopy(_raw_config(name))
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
    for name in [*MESHES, "cavity"]
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


@pytest.mark.parametrize(
    ("name", "mode"), [(n, m) for n, m in OUTER_CASES if n != "320x120"]
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
    the vent faces, and so does the core's outer iteration.
    """
    raw = _raw("80x30-vents-shelf", "coupled", max_simple_iter=3)
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
        messages.append((str(refused.value), len(solver.residual_history)))
    assert messages[1] == messages[0]
    assert "vent faces" in messages[0][0]


def _log_lines(records: list[logging.LogRecord]) -> list[tuple[str, str]]:
    return [
        (r.levelname, r.getMessage().split(" = ")[0])
        for r in records
        if r.name == "src.solver_staggered"
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
