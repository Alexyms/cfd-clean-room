"""VAL-008: grid convergence of the limited transport scheme (REQ-N02; ADR-014 E).

The observed order of the scheme's advection under refinement: the relative L2
error against the exact translation on a sequence of grids, and the order
between a pair as log2 of the error ratio. Three rows, as ADR-014 section E
states them and its table T5 measured them (docs/reports/probe59/order59.py):

- The smooth monotone front, the criterion row. A tanh front of width 0.2 m
  travels 3 widths (0.6 m) at the supply speed on the 2.0 by 0.4 m channel,
  starting 4.5 widths from the inlet, with the Courant number halved per
  refinement. Asserted: the order between the grids is within 0.2 of 2. Upwind
  is the control, asserted within 0.2 of 1 on the same rows.
- The Gaussian pulse. Reported, no threshold: the limiter is first order at an
  extremum (ADR-011 B), so its order is between the two.
- The front on a stretched mesh. Reported, no threshold (issue 53 D6): the
  limiter's three uniform-mesh terms on a non-uniform axis.

The Courant number is halved with the spacing so the forward Euler error falls
as dx^2 and the space scheme's order is what shows. At a fixed Courant number
the time error is O(dx) and the observed order tends to 1 whatever the scheme
(ADR-014 E, "Why not the pulse at a fixed Courant number").

Grids. The criterion row runs the 0.02 and 0.01 m grids, the two section E
names: the coarser holds 10 cells across the front's width, the least the rule
allows (the 0.04 m grid holds 5 and reads 1.64 in T5). Both rows together take
about 15 s. Adding the 0.005 m grid reads 1.90 between it and 0.01 m, as T5
records, and takes about 93 s alone, which is why this file stops at 0.01 m;
ADR-014 E's as-built note holds that reading.

Instrument checks (marked unit) pin the exact front averages, the order
formula, the Courant rule and the 10-cell rule. The planted defects of the
build are kept as tests: the limiter forced to upwind, the Courant number held
fixed, and the exact solution shifted by one cell each push the criterion row
out of its band.
"""

import math
from dataclasses import dataclass
from functools import cache
from itertools import pairwise
from time import perf_counter

import numpy as np
import pytest

import src.scalar_scheme as scalar_scheme
from src.config import UMIST, UPWIND
from src.mesh import Mesh
from src.solver_transport import TransportSolver
from validation.metrics import relative_l2
from validation.transport_cases import (
    CFL_NUMBER,
    SUPPLY_SPEED,
    FixedConditions,
    ScalarPhysics,
    gaussian_cell_averages,
    transport_config,
    uniform_face_field,
    zero_conditions,
)

# ADR-014 E, T5's wide front: 3 widths of travel, the front 4.5 widths from the
# inlet at the start, on the 2.0 by 0.4 m channel at the supply speed.
CHANNEL_WIDTH = 2.0
CHANNEL_HEIGHT = 0.4
FRONT_WIDTH = 0.2
FRONT_TRAVEL = 0.6
FRONT_START = 0.9
# The pulse of T5's aligned case.
PULSE_SIGMA = 0.1
PULSE_TRAVEL = 1.0
PULSE_START = 0.45
# Refinement level k has spacing REFERENCE_SPACING / 2^k and Courant number
# CFL_NUMBER / 2^k, so level 0 is T5's coarsest grid at the product case's 0.1.
REFERENCE_SPACING = 0.04
CRITERION_LEVELS = (1, 2)
REPORTED_LEVELS = (0, 1, 2)
# The least cells across the front's width on the coarser grid of a pair.
MIN_CELLS_ACROSS_WIDTH = 10
THEORETICAL_ORDER = 2.0
UPWIND_ORDER = 1.0
ORDER_TOLERANCE = 0.2
# The stretched row clusters x toward the inlet and the outlet so the end cells
# are this fraction of the uniform spacing and the cells widen toward the
# centre, where the front travels. The ratio between adjacent widths then falls
# with refinement, as it does on a mesh mapped smoothly from a uniform one; a
# fixed per-cell ratio would stop the centre refining.
WALL_SPACING_FRACTION = 0.5
# The widest cell of the stretched mesh over the narrowest, at least: a mesh
# that silently came back uniform reads 1.
MIN_STRETCH = 3.0

FRONT = "front"
PULSE = "pulse"


@dataclass(frozen=True)
class Case:
    """One advected profile: what moves, from where, how far, on what mesh.

    Parameters
    ----------
    name : str
        Label printed with the row.
    profile : str
        ``FRONT`` or ``PULSE``.
    start : float
        x of the profile's centre at t = 0, in metres.
    travel : float
        Distance the profile moves along x, in metres.
    stretched : bool
        True clusters the x axis toward the inlet and the outlet.
    """

    name: str
    profile: str
    start: float
    travel: float
    stretched: bool = False


FRONT_CASE = Case("front", FRONT, FRONT_START, FRONT_TRAVEL)
PULSE_CASE = Case("pulse", PULSE, PULSE_START, PULSE_TRAVEL)
STRETCHED_FRONT_CASE = Case("stretched front", FRONT, FRONT_START, FRONT_TRAVEL, True)


@dataclass(frozen=True)
class Run:
    """One advection to the end of a case on one grid.

    Parameters
    ----------
    level : int
        Refinement level, spacing ``REFERENCE_SPACING / 2^level``.
    mesh : Mesh
        The grid's mesh.
    field : np.ndarray
        The final cell field, [ny, nx].
    steps : int
        Time steps taken.
    seconds : float
        Wall time of the steps.
    """

    level: int
    mesh: Mesh
    field: np.ndarray
    steps: int
    seconds: float


def spacing_at(level: int) -> float:
    """Uniform cell spacing at a refinement level, in metres."""
    return REFERENCE_SPACING / 2**level


def courant_at(level: int) -> float:
    """Courant number at a refinement level: halved with each refinement."""
    return CFL_NUMBER / 2**level


def cells_across_width(level: int) -> float:
    """Cells across the front's width at a refinement level."""
    return FRONT_WIDTH / spacing_at(level)


def front_cell_averages(
    x_faces: np.ndarray, y_faces: np.ndarray, centre: float, width: float
) -> np.ndarray:
    """Exact cell averages of 0.5 (1 + tanh((x - centre) / width)), uniform in y.

    Parameters
    ----------
    x_faces, y_faces : np.ndarray
        Face coordinates, ``mesh.x`` and ``mesh.y``.
    centre : float
        x of the front's midpoint, in metres.
    width : float
        The tanh length scale, in metres.

    Returns
    -------
    np.ndarray
        [ny, nx], C-contiguous. Each column holds the average of the profile
        over the cell's x extent.

    Notes
    -----
    The antiderivative is 0.5 (s + width log cosh((s - centre) / width)), with
    log cosh written as |z| + log1p(exp(-2|z|)) - log 2 so a large argument
    does not overflow.
    """

    def primitive(s: np.ndarray) -> np.ndarray:
        z = (s - centre) / width
        log_cosh = np.abs(z) + np.log1p(np.exp(-2.0 * np.abs(z))) - np.log(2.0)
        return 0.5 * (s + width * log_cosh)

    column = (primitive(x_faces[1:]) - primitive(x_faces[:-1])) / np.diff(x_faces)
    return np.ascontiguousarray(
        np.broadcast_to(column[None, :], (y_faces.size - 1, x_faces.size - 1))
    )


def profile_at(case: Case, mesh: Mesh, centre: float) -> np.ndarray:
    """The case's profile as exact cell averages with its centre at ``centre``."""
    if case.profile == FRONT:
        return front_cell_averages(mesh.x, mesh.y, centre, FRONT_WIDTH)
    return gaussian_cell_averages(
        mesh.x, mesh.y, (centre, CHANNEL_HEIGHT / 2.0), PULSE_SIGMA
    )


def observed_order(coarse_error: float, fine_error: float) -> float:
    """Observed order between two grids of spacing ratio 2: log2 of the error ratio.

    Parameters
    ----------
    coarse_error, fine_error : float
        Errors on the coarser and the finer grid, both positive.

    Returns
    -------
    float
        ``log2(coarse_error / fine_error)``.

    Raises
    ------
    ValueError
        If either error is not positive and finite.
    """
    for error in (coarse_error, fine_error):
        if not (math.isfinite(error) and error > 0.0):
            raise ValueError(f"an error must be positive and finite, got {error}")
    return math.log2(coarse_error / fine_error)


def require_resolved(levels: tuple[int, ...]) -> None:
    """Refuse a grid sequence whose coarser grid of a pair is under-resolved.

    Parameters
    ----------
    levels : tuple[int, ...]
        Refinement levels, coarse to fine, each one finer than the last.

    Raises
    ------
    ValueError
        If the levels are not consecutive and rising, or any pair's coarser
        grid holds fewer than ``MIN_CELLS_ACROSS_WIDTH`` cells across the
        front's width (ADR-014 E: the 5-cell pair reads 1.64 in T5).
    """
    if len(levels) < 2 or any(b - a != 1 for a, b in pairwise(levels)):
        raise ValueError(
            f"levels must be at least two consecutive levels, got {levels}"
        )
    for coarse in levels[:-1]:
        cells = cells_across_width(coarse)
        if round(cells, 9) < MIN_CELLS_ACROSS_WIDTH:
            raise ValueError(
                f"level {coarse} holds {cells:g} cells across the front's width, "
                f"under the {MIN_CELLS_ACROSS_WIDTH} the rule needs"
            )


def within_band(order: float, target: float) -> bool:
    """True when the order is within ``ORDER_TOLERANCE`` of the target."""
    return abs(order - target) <= ORDER_TOLERANCE


def advance(case: Case, level: int, scheme: str, courant: float | None = None) -> Run:
    """Advect a case's profile to the end of its travel on one grid.

    Parameters
    ----------
    case : Case
        The profile, its start and its travel.
    level : int
        Refinement level of the grid.
    scheme : str
        ``UMIST`` or ``UPWIND``.
    courant : float, optional
        Courant number; None is ``courant_at(level)``, halved per refinement.

    Returns
    -------
    Run
        The final field after the fewest equal steps that keep each at or below
        ``stable_dt``.
    """
    spacing = spacing_at(level)
    nx, ny = round(CHANNEL_WIDTH / spacing), round(CHANNEL_HEIGHT / spacing)
    mesh_section = (
        {"x": {"min_wall_spacing": WALL_SPACING_FRACTION * spacing}}
        if case.stretched
        else None
    )
    config = transport_config(
        CHANNEL_WIDTH,
        CHANNEL_HEIGHT,
        nx,
        ny,
        cfl_number=courant_at(level) if courant is None else courant,
        scheme=scheme,
        mesh=mesh_section,
    )
    mesh = Mesh(config)
    faces = uniform_face_field(mesh, SUPPLY_SPEED, 0.0)
    solver = TransportSolver(
        mesh,
        config,
        ScalarPhysics(settling=0.0, diffusion=0.0),
        FixedConditions(zero_conditions(mesh)),
    )
    t_end = case.travel / SUPPLY_SPEED
    steps = math.ceil(t_end / solver.stable_dt(faces, 0))
    dt = t_end / steps
    field = profile_at(case, mesh, case.start)
    start = perf_counter()
    for _ in range(steps):
        field = solver.solve_timestep(field, faces, 0, dt)
    return Run(level, mesh, field, steps, perf_counter() - start)


@cache
def halved_run(case: Case, level: int, scheme: str) -> Run:
    """``advance`` at the halved Courant number, kept so each run is made once."""
    return advance(case, level, scheme)


def error_of(run: Run, case: Case, shift: float = 0.0) -> float:
    """Relative L2 error of a run against the exact translation.

    Parameters
    ----------
    run : Run
        The run to score.
    case : Case
        Supplies the profile, its start and its travel.
    shift : float
        Added to the exact profile's centre, in metres; zero except where a
        test plants a displaced exact solution.

    Returns
    -------
    float
        ``relative_l2`` of the final field against the profile centred at
        ``start + U t_end + shift``.
    """
    t_end = case.travel / SUPPLY_SPEED
    centre = case.start + SUPPLY_SPEED * t_end + shift
    return relative_l2(run.field, profile_at(case, run.mesh, centre), run.mesh)


def orders_of(errors: list[float]) -> list[float]:
    """The observed order between each successive pair of errors."""
    return [observed_order(a, b) for a, b in pairwise(errors)]


def report_row(label: str, runs: list[Run], errors: list[float]) -> list[float]:
    """Print a row's errors, orders and run times; return the orders."""
    orders = orders_of(errors)
    spacings = ", ".join(f"{spacing_at(r.level):g}" for r in runs)
    print(
        f"VAL-008 {label}: dx {spacings} m; relative L2 "
        f"{', '.join(f'{e:.4e}' for e in errors)}; orders "
        f"{', '.join(f'{o:.2f}' for o in orders)}; steps "
        f"{', '.join(str(r.steps) for r in runs)}; "
        f"{sum(r.seconds for r in runs):.1f} s"
    )
    return orders


def row(
    case: Case, scheme: str, levels: tuple[int, ...]
) -> tuple[list[Run], list[float]]:
    """The runs and the errors of a case and scheme over a level sequence."""
    runs = [halved_run(case, level, scheme) for level in levels]
    return runs, [error_of(run, case) for run in runs]


# ---------------------------------------------------------------------------
# The instrument
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestInstrument:
    """The exact solution, the order formula and the rules the row is read by."""

    def test_front_averages_equal_a_fine_midpoint_quadrature(self) -> None:
        """The closed form is the mean of the tanh profile over each cell."""
        x = np.array([0.0, 0.07, 0.19, 0.5, 0.52, 1.1])
        y = np.array([0.0, 0.1, 0.4])
        averages = front_cell_averages(x, y, 0.3, 0.2)
        for i in range(x.size - 1):
            sub = np.linspace(x[i], x[i + 1], 200_001)
            mid = 0.5 * (sub[1:] + sub[:-1])
            expected = np.mean(0.5 * (1.0 + np.tanh((mid - 0.3) / 0.2)))
            assert averages[0, i] == pytest.approx(expected, abs=1e-12)

    def test_front_averages_are_uniform_in_y_and_laid_out_for_the_solver(self) -> None:
        """Shape [ny, nx], float64, C-contiguous, every row equal."""
        x, y = np.linspace(0.0, 2.0, 21), np.linspace(0.0, 0.4, 5)
        averages = front_cell_averages(x, y, 0.9, FRONT_WIDTH)
        assert averages.shape == (4, 20)
        assert averages.dtype == np.float64
        assert averages.flags["C_CONTIGUOUS"]
        assert np.all(averages == averages[0])

    def test_front_averages_survive_a_large_argument(self) -> None:
        """A front far from the cell does not overflow log cosh."""
        x, y = np.array([0.0, 1.0e4]), np.array([0.0, 1.0])
        averages = front_cell_averages(x, y, 5.0e3, 0.2)
        assert np.all(np.isfinite(averages))
        assert averages[0, 0] == pytest.approx(0.5, abs=1e-12)

    def test_the_front_starts_clear_of_the_inlet(self) -> None:
        """Section E: 4.5 widths from the inlet, a tail under 2e-4 of the height.

        A larger tail would let the inlet's zero cut the profile and floor
        the error, which the first placing of the wide front showed.
        """
        assert math.isclose(FRONT_START, 4.5 * FRONT_WIDTH)
        assert math.isclose(FRONT_TRAVEL, 3.0 * FRONT_WIDTH)
        tail = 0.5 * (1.0 + math.tanh((0.0 - FRONT_START) / FRONT_WIDTH))
        assert tail < 2.0e-4
        mesh = Mesh(transport_config(CHANNEL_WIDTH, CHANNEL_HEIGHT, 100, 20))
        initial = profile_at(FRONT_CASE, mesh, FRONT_START)
        assert initial[0, 0] < 2.0e-4

    def test_the_front_ends_inside_the_channel(self) -> None:
        """The front's tail reaches the outlet at under 1e-2 of its height."""
        end = FRONT_START + FRONT_TRAVEL
        assert end + 2.5 * FRONT_WIDTH <= CHANNEL_WIDTH
        outlet_gap = 1.0 - 0.5 * (1.0 + math.tanh((CHANNEL_WIDTH - end) / FRONT_WIDTH))
        assert outlet_gap < 1.0e-2

    @pytest.mark.parametrize(
        ("coarse", "fine", "expected"),
        [(4.0, 1.0, 2.0), (2.0, 1.0, 1.0), (1.0, 1.0, 0.0), (1.0, 2.0, -1.0)],
    )
    def test_observed_order_is_log2_of_the_error_ratio(
        self, coarse: float, fine: float, expected: float
    ) -> None:
        """A quartering is order 2, a halving order 1, no change order 0."""
        assert observed_order(coarse, fine) == pytest.approx(expected)

    @pytest.mark.parametrize(
        ("coarse", "fine"),
        [(0.0, 1.0), (1.0, 0.0), (-1.0, 1.0), (math.nan, 1.0), (1.0, math.inf)],
    )
    def test_observed_order_refuses_an_error_that_is_not_positive_and_finite(
        self, coarse: float, fine: float
    ) -> None:
        """A zero or non-finite error has no logarithm worth reporting."""
        with pytest.raises(ValueError, match="positive and finite"):
            observed_order(coarse, fine)

    def test_the_courant_number_and_the_spacing_halve_together(self) -> None:
        """Level k is REFERENCE_SPACING / 2^k at CFL_NUMBER / 2^k, so dt falls as dx^2."""
        assert [spacing_at(k) for k in range(4)] == [0.04, 0.02, 0.01, 0.005]
        assert [courant_at(k) for k in range(4)] == [0.1, 0.05, 0.025, 0.0125]
        assert courant_at(0) == CFL_NUMBER
        for k in range(3):
            assert courant_at(k + 1) / courant_at(k) == 0.5
            assert spacing_at(k + 1) / spacing_at(k) == 0.5

    def test_the_criterion_grids_hold_ten_and_twenty_cells_across_the_width(
        self,
    ) -> None:
        """The pair's coarser grid is at the rule's limit, the finer one twice it."""
        assert [cells_across_width(k) for k in (0, 1, 2, 3)] == pytest.approx(
            [5.0, 10.0, 20.0, 40.0]
        )
        require_resolved(CRITERION_LEVELS)
        require_resolved((1, 2, 3))

    def test_a_pair_with_a_coarser_grid_under_ten_cells_is_refused(self) -> None:
        """The 5-cell grid cannot be the coarser grid of a pair (T5: 1.64)."""
        with pytest.raises(ValueError, match="under the 10"):
            require_resolved((0, 1))
        with pytest.raises(ValueError, match="under the 10"):
            require_resolved((0, 1, 2))

    @pytest.mark.parametrize("levels", [(), (1,), (1, 3), (2, 1), (1, 1)])
    def test_a_sequence_that_is_not_consecutive_refinement_is_refused(
        self, levels: tuple[int, ...]
    ) -> None:
        """The order formula assumes a spacing ratio of 2 between neighbours."""
        with pytest.raises(ValueError, match="consecutive"):
            require_resolved(levels)

    @pytest.mark.parametrize(
        ("order", "target", "expected"),
        [
            (1.8 + 1e-9, 2.0, True),
            (1.8 - 1e-6, 2.0, False),
            (2.2 - 1e-9, 2.0, True),
            (2.2 + 1e-6, 2.0, False),
            (0.8 + 1e-9, 1.0, True),
            (1.2 + 1e-6, 1.0, False),
        ],
    )
    def test_the_band_is_a_fifth_either_side(
        self, order: float, target: float, expected: bool
    ) -> None:
        """Within 0.2 of the target, on both sides."""
        assert within_band(order, target) is expected

    def test_a_synthetic_second_order_error_sequence_reads_two(self) -> None:
        """Errors c h^2 through the pipeline's own formula give 2, and c h gives 1."""
        h = [spacing_at(k) for k in (1, 2, 3)]
        assert orders_of([7.0 * s**2 for s in h]) == pytest.approx([2.0, 2.0])
        assert orders_of([7.0 * s for s in h]) == pytest.approx([1.0, 1.0])


# ---------------------------------------------------------------------------
# The rows
# ---------------------------------------------------------------------------


@pytest.mark.validation
def test_front_converges_at_second_order_val008() -> None:
    """VAL-008 criterion row: the front's observed order is within 0.2 of 2.

    Grids 0.02 and 0.01 m (10 and 20 cells across the width), the Courant
    number 0.05 and 0.025. T5 records 1.83 between them.
    """
    require_resolved(CRITERION_LEVELS)
    runs, errors = row(FRONT_CASE, UMIST, CRITERION_LEVELS)
    orders = report_row("front, UMIST", runs, errors)
    for order in orders:
        assert within_band(order, THEORETICAL_ORDER), orders


@pytest.mark.validation
def test_upwind_control_reads_first_order_on_the_front_val008() -> None:
    """VAL-008 control: upwind on the same front reads within 0.2 of 1.

    The same grids, Courant numbers and error metric read first order for a
    first-order scheme, so the criterion's reading near 2 is the limited
    scheme's and not the instrument's. A scheme that silently fell back to
    upwind reads here what this row reads and fails the criterion row. T5
    records 0.86 between the grids.
    """
    require_resolved(CRITERION_LEVELS)
    runs, errors = row(FRONT_CASE, UPWIND, CRITERION_LEVELS)
    orders = report_row("front, upwind", runs, errors)
    for order in orders:
        assert within_band(order, UPWIND_ORDER), orders


@pytest.mark.validation
def test_gaussian_pulse_order_is_reported_val008() -> None:
    """VAL-008 reported row: the pulse's orders on 0.04, 0.02 and 0.01 m, no threshold.

    The limiter is first order at the pulse's extremum, so the order is
    between 1 and 2 (T5: 1.17 and 1.15 on these grids). Only that the errors
    are finite and positive is asserted.
    """
    runs, errors = row(PULSE_CASE, UMIST, REPORTED_LEVELS)
    orders = report_row("pulse, UMIST", runs, errors)
    assert len(orders) == len(REPORTED_LEVELS) - 1
    assert all(math.isfinite(e) and e > 0.0 for e in errors)


@pytest.mark.validation
def test_front_on_the_stretched_mesh_is_reported_val008() -> None:
    """VAL-008 reported row: the front on an x-clustered mesh, no threshold.

    Issue 53 D6: the limiter's clamp is applied to the quadratic QUICK value
    with its other three terms in their uniform-mesh form. T5 has no front on
    a stretched mesh, so the figures are new. Asserted: the mesh is stretched,
    and the errors are finite and positive. The error is
    ``validation.metrics.relative_l2``, an unweighted norm over the cells
    (test 65 S1): on this mesh the end cells are 0.50 dx wide and the
    centre's about 1.75 dx, so it is not the area-weighted norm, and the
    row's figures are not comparable with an area-weighted one.
    """
    runs, errors = row(STRETCHED_FRONT_CASE, UMIST, REPORTED_LEVELS)
    for run in runs:
        widths = np.diff(run.mesh.x)
        assert widths.max() / widths.min() >= MIN_STRETCH
    orders = report_row("front on the stretched mesh, UMIST", runs, errors)
    assert len(orders) == len(REPORTED_LEVELS) - 1
    assert all(math.isfinite(e) and e > 0.0 for e in errors)


# ---------------------------------------------------------------------------
# The planted defects
# ---------------------------------------------------------------------------


@pytest.mark.validation
def test_limiter_forced_to_upwind_fails_the_criterion_and_not_the_control_val008(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Plant: the limiter returns the upstream cell, so UMIST is upwind in effect.

    The criterion row leaves its band, which is what the row exists to catch.
    The control does not move, because upwind never calls the limiter.
    """
    monkeypatch.setattr(
        scalar_scheme, "limited_face_values", lambda c_up, c_c, c_d, quick: c_c
    )
    planted = [advance(FRONT_CASE, level, UMIST) for level in CRITERION_LEVELS]
    orders = orders_of([error_of(run, FRONT_CASE) for run in planted])
    print(f"VAL-008 plant, limiter forced to upwind: orders {orders[0]:.2f}")
    assert not any(within_band(order, THEORETICAL_ORDER) for order in orders)
    # Upwind never calls the limiter, so a run cached under this patch is the
    # run without it.
    _, control_errors = row(FRONT_CASE, UPWIND, CRITERION_LEVELS)
    assert all(within_band(o, UPWIND_ORDER) for o in orders_of(control_errors))


@pytest.mark.validation
def test_courant_number_held_fixed_fails_the_criterion_val008() -> None:
    """Plant: the Courant number stays at its coarse value, so dt falls only as dx.

    The forward Euler error then falls as dx and the order falls toward 1
    whatever the space scheme (section E, "why not").
    """
    planted = [
        advance(FRONT_CASE, level, UMIST, CFL_NUMBER) for level in CRITERION_LEVELS
    ]
    orders = orders_of([error_of(run, FRONT_CASE) for run in planted])
    print(f"VAL-008 plant, Courant number not halved: orders {orders[0]:.2f}")
    assert not any(within_band(order, THEORETICAL_ORDER) for order in orders)


@pytest.mark.validation
def test_exact_front_shifted_by_one_cell_fails_the_criterion_val008() -> None:
    """Plant: the exact translation is one cell of the grid too far.

    The shift adds an error proportional to the spacing, so the order leaves
    its band even though the solver is unchanged.
    """
    runs = [halved_run(FRONT_CASE, level, UMIST) for level in CRITERION_LEVELS]
    errors = [error_of(run, FRONT_CASE, shift=spacing_at(run.level)) for run in runs]
    orders = orders_of(errors)
    print(f"VAL-008 plant, exact front shifted one cell: orders {orders[0]:.2f}")
    assert not any(within_band(order, THEORETICAL_ORDER) for order in orders)
