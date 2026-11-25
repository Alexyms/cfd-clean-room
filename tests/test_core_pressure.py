"""The compiled core's pressure correction against the NumPy reference (REQ-N03 level 1, ADR-017 D).

``_cfdcore.PressureCorrector2`` against ``PressureCorrector`` and ``Multigrid2``
against ``Multigrid``, given the same inputs, by the kind of each block (ADR-017
G). Bit for bit: the coefficients, the imbalance, the right-hand side and its
projection, the operator, every level's arrays, a sweep, the group sums, the
coarsest matrix and the correction given p'. The coarsest level's pseudo-inverse
against one in 60-digit decimals, on the scale of its condition, as NumPy's is; a
whole V-cycle to 1e-12. At the reference's iteration count, the
fields to 1e-12: the conjugate gradients under both preconditioners, and bit for
bit against the reference's own loop handed the core's dot product and cycle.
On prompt 87's meshes, the vents', the cavity's and small rooms of every row
length, with planted SOLID cells and outlets. Then the refusals. Skipped with its
reason when ``_cfdcore`` cannot be imported, an error instead with
``CFD_CORE_REQUIRED=1``, as ``tests/test_core_parity.py``.
"""

import contextlib
import dataclasses
import functools
import os
import warnings
from collections.abc import Callable, Iterator
from types import SimpleNamespace

import numpy as np
import pytest

from src import multigrid, pressure
from src.boundary_staggered import StaggeredBoundary
from src.config import MULTIGRID, PRESSURE_JACOBI, SimConfig
from src.mesh import BOUNDARY, FLUID, SOLID, Mesh
from src.momentum import MomentumPrediction
from src.multigrid import Multigrid
from src.pressure import PressureCoefficients, PressureCorrector
from tests import core_mapping, decimal_inverse
from validation.cases import load_case

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

# The parity meshes, after the skip above: that module needs the core too.
from tests.test_core_parity import (
    ALL_MESHES,
    _config,
    _fortran,
    _mesh,
    _strided,
    _unaligned_like,
)

pytestmark = [pytest.mark.parity, pytest.mark.integration]

SEED = 93
TOLERANCE = 1e-12
FIVE = ("a_p", "a_e", "a_w", "a_n", "a_s")
FIELDS = ("u", "v", "p", "p_prime")
CAVITY = "cavity40"
NAMES = [*ALL_MESHES, CAVITY]
EDGES = ("left", "right", "bottom", "top")
LEVELS = core_mapping.MULTIGRID_LEVELS
# What a planted momentum diagonal is at the faces it is an unknown's: values
# within a decade of one another as a room's are, or over a range of decades.
SPREADS = {"decade": 1.0, "decades": 3.0}


# ---- The two correctors ------------------------------------------------------


@dataclasses.dataclass(frozen=True, eq=False)
class Room:
    """A mesh with its outlets, and the two correctors built on it."""

    mesh: object
    config: object
    reference: PressureCorrector
    core: "_cfdcore.PressureCorrector2"


class PlantedBoundary:
    """What a corrector reads of a boundary, with planted outlets.

    Parameters
    ----------
    outlets : dict[str, np.ndarray]
        The outlet faces of each edge, bool.
    inflow : float
        The total inlet flux, which the floor's flux scale is made from.
    """

    def __init__(self, outlets: dict[str, np.ndarray], inflow: float = 0.3) -> None:
        self._outlets = outlets
        self._inflow = inflow

    def pressure_outlets(self) -> dict[str, SimpleNamespace]:
        """The planted outlets, by edge."""
        return {e: SimpleNamespace(is_outlet=o) for e, o in self._outlets.items()}

    def has_pressure_outlet(self) -> bool:
        """Whether any is planted."""
        return any(bool(o.any()) for o in self._outlets.values())

    def get_total_inlet_flux(self) -> float:
        """The planted inflow."""
        return self._inflow

    def get_max_boundary_velocity(self) -> float:
        """A velocity for a room without inflow."""
        return 1.0


def _room(mesh: object, config: object, boundary: object) -> Room:
    reference = PressureCorrector(mesh, config, boundary)
    core = core_mapping.pressure_corrector(mesh, config, boundary, reference)
    return Room(mesh, config, reference, core)


def _edge_lengths(mesh: object) -> dict[str, int]:
    ny, nx = mesh.cell_type.shape
    return {"left": ny, "right": ny, "bottom": nx, "top": nx}


def _no_outlets(mesh: object) -> dict[str, np.ndarray]:
    return {e: np.zeros(n, dtype=bool) for e, n in _edge_lengths(mesh).items()}


@functools.cache
def _named_mesh(name: str) -> Mesh:
    return Mesh(load_case("cavity")) if name == CAVITY else _mesh(name)


def _solver_config(name: str, **keys: object) -> SimpleNamespace:
    """What a corrector reads of a configuration, with some of it replaced."""
    config: SimConfig = load_case("cavity") if name == CAVITY else _config(name)
    read = {
        "rho": config.rho,
        "alpha_pressure": config.alpha_pressure,
        "max_pressure_iter": config.max_pressure_iter,
        "pressure_rtol": config.pressure_rtol,
        "pressure_preconditioner": config.pressure_preconditioner,
    }
    return SimpleNamespace(**{**read, **keys})


@functools.cache
def _closed(name: str, preconditioner: str = MULTIGRID) -> Room:
    """A named mesh with its own boundary: none of them has a pressure outlet."""
    mesh = _named_mesh(name)
    config = load_case("cavity") if name == CAVITY else _config(name)
    return _room(
        mesh,
        _solver_config(name, pressure_preconditioner=preconditioner),
        StaggeredBoundary(mesh, config),
    )


@functools.cache
def _open(name: str, preconditioner: str = MULTIGRID) -> Room:
    """A named mesh with outlets planted on all four edges."""
    mesh = _named_mesh(name)
    rng = np.random.default_rng([SEED, NAMES.index(name)])
    outlets = {e: rng.random(n) < 0.2 for e, n in _edge_lengths(mesh).items()}
    return _room(
        mesh,
        _solver_config(name, pressure_preconditioner=preconditioner),
        PlantedBoundary(outlets),
    )


def _small_room(
    ny: int,
    nx: int,
    rng: np.random.Generator,
    opened: bool,
    preconditioner: str = MULTIGRID,
    solid: float = 0.15,
    rtol: float = 1e-9,
    max_iter: int = 400,
) -> Room:
    """A small grid of random widths with SOLID cells and outlets the reference accepts."""
    for _attempt in range(200):
        widths_x, widths_y = 0.5 + rng.random(nx), 0.5 + rng.random(ny)
        x = np.concatenate(([0.0], np.cumsum(widths_x)))
        y = np.concatenate(([0.0], np.cumsum(widths_y)))
        cell_type = np.full((ny, nx), FLUID, dtype=np.int32)
        cell_type[rng.random((ny, nx)) < solid] = SOLID
        # BOUNDARY cells are open: their rows come before the first FLUID one.
        cell_type[0][cell_type[0] != SOLID] = BOUNDARY if ny > 1 else FLUID
        mesh = SimpleNamespace(
            x=x,
            xc=0.5 * (x[:-1] + x[1:]),
            y=y,
            yc=0.5 * (y[:-1] + y[1:]),
            dx_cell=np.diff(x),
            dy_cell=np.diff(y),
            cell_type=cell_type,
        )
        outlets = _no_outlets(mesh)
        if opened:
            outlets = {e: rng.random(n) < 0.6 for e, n in _edge_lengths(mesh).items()}
        config = SimpleNamespace(
            rho=1.2,
            alpha_pressure=0.3,
            max_pressure_iter=max_iter,
            pressure_rtol=rtol,
            pressure_preconditioner=preconditioner,
        )
        try:
            return _room(mesh, config, PlantedBoundary(outlets))
        except ValueError:
            continue
    raise AssertionError("no small room the reference accepts")


# ---- The planted inputs ------------------------------------------------------


@dataclasses.dataclass
class Inputs:
    """The arguments of one correction."""

    u_star: np.ndarray
    v_star: np.ndarray
    a_p_u: np.ndarray
    a_p_v: np.ndarray
    p: np.ndarray

    @property
    def prediction(self) -> MomentumPrediction:
        """The reference's first argument."""
        return MomentumPrediction(self.u_star, self.v_star, self.a_p_u, self.a_p_v)

    @property
    def stars(self) -> tuple[np.ndarray, np.ndarray]:
        """u* and v*."""
        return self.u_star, self.v_star

    @property
    def diagonals(self) -> tuple[np.ndarray, np.ndarray]:
        """The two diagonals."""
        return self.a_p_u, self.a_p_v

    @property
    def arguments(self) -> tuple[np.ndarray, ...]:
        """The core's arguments."""
        return *self.stars, *self.diagonals, self.p


def _planted(
    room: Room, rng: np.random.Generator, spread: float = 1.0, faces: str = "unknowns"
) -> Inputs:
    """Planted inputs of one correction.

    Parameters
    ----------
    room : Room
        Supplies the mesh.
    rng : np.random.Generator
        Draws the values.
    spread : float
        The decades the diagonals range over.
    faces : str
        "unknowns" puts a diagonal at every face two open cells share, as a
        momentum prediction does; "some" takes a share of those away as
        zeros, negative values and NaNs, as faces that are not correctable
        are; "any" plants one on edge faces and faces of SOLID cells too.
    """
    solid = room.mesh.cell_type == SOLID
    ny, nx = solid.shape
    inner_u = np.zeros((ny, nx + 1), dtype=bool)
    inner_v = np.zeros((ny + 1, nx), dtype=bool)
    inner_u[:, 1:-1] = ~(solid[:, 1:] | solid[:, :-1])
    inner_v[1:-1, :] = ~(solid[1:, :] | solid[:-1, :])
    out = []
    for inner in (inner_u, inner_v):
        diagonal = 10.0 ** rng.uniform(-spread / 2, spread / 2, inner.shape) * 0.02
        if faces != "any":
            diagonal = np.where(inner, diagonal, 0.0)
        if faces == "some":
            spoilt = rng.random(inner.shape) < 0.2
            values = rng.choice([0.0, -0.0, -0.01, np.nan], size=inner.shape)
            diagonal = np.where(spoilt, values, diagonal)
        out.append(diagonal)
    u = rng.standard_normal(inner_u.shape)
    v = rng.standard_normal(inner_v.shape)
    if faces != "any":
        u[:, 1:-1][~inner_u[:, 1:-1]] = 0.0
        v[1:-1, :][~inner_v[1:-1, :]] = 0.0
    u[rng.random(u.shape) < 0.02] = -0.0
    p = rng.standard_normal((ny, nx)) * 10.0 ** rng.uniform(-3, 3, (ny, nx))
    return Inputs(u, v, *out, p)


def _system(
    room: Room, inputs: Inputs
) -> tuple[PressureCoefficients, list[np.ndarray]]:
    """The reference's coefficients, as its object and as the core's five arrays."""
    with np.errstate(all="ignore"):
        c = room.reference.coefficients(*inputs.diagonals)
    return c, [getattr(c, name) for name in FIVE]


def _differ(a: np.ndarray, b: np.ndarray) -> int:
    """How many entries of two float64 arrays differ in their 64 bits; -1 for shapes."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return -1
    a, b = np.ascontiguousarray(a), np.ascontiguousarray(b)
    return int(np.count_nonzero(a.view(np.uint64) != b.view(np.uint64)))


def _largest(a: np.ndarray, reference: np.ndarray) -> float:
    """The largest difference over the reference's largest magnitude."""
    scale = float(np.abs(reference).max())
    difference = float(np.abs(a - reference).max())
    return difference / scale if scale > 0.0 else difference


class _HandedError(Exception):
    """Leaves ``correct`` once its solve's arguments are known."""


def _handed(room: Room, inputs: Inputs) -> np.ndarray:
    """The right-hand side ``correct`` hands the conjugate gradients."""
    kept: dict[str, np.ndarray] = {}

    def stand_in(
        _apply: object, _precondition: object, f: np.ndarray, *_numbers: object
    ) -> None:
        kept["f"] = f.copy()
        raise _HandedError

    with pytest.MonkeyPatch.context() as patch, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        patch.setattr(pressure, "conjugate_gradient", stand_in)
        patch.setattr(pressure, "multigrid_preconditioner", lambda *_a: None)
        with pytest.raises(_HandedError):
            room.reference.correct(inputs.prediction, inputs.p)
    return kept["f"]


def _corrected_by(
    room: Room, inputs: Inputs, solved: np.ndarray
) -> pressure.PressureCorrection:
    """The reference's correction by a given solution of the p' equation."""

    def stand_in(*_args: object) -> pressure.ConjugateGradientResult:
        return pressure.ConjugateGradientResult(solved.copy(), 0, False, 0.0, 0)

    with pytest.MonkeyPatch.context() as patch, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        patch.setattr(pressure, "conjugate_gradient", stand_in)
        patch.setattr(pressure, "multigrid_preconditioner", lambda *_a: None)
        return room.reference.correct(inputs.prediction, inputs.p)


def _dense_matrix(c: PressureCoefficients) -> np.ndarray:
    """The matrix the reference hands ``np.linalg.pinv`` for a coarsest level."""
    kept: dict[str, np.ndarray] = {}

    def stand_in(matrix: np.ndarray, hermitian: bool = False) -> np.ndarray:
        kept["matrix"] = matrix.copy()
        return matrix

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(np.linalg, "pinv", stand_in)
        multigrid._dense_pseudo_inverse(c)
    return kept["matrix"]


def _rooms(name: str) -> list[Room]:
    return [_closed(name), _open(name)]


# ---- The first kind, bit for bit ---------------------------------------------


def _assert_equation(room: Room, inputs: Inputs) -> None:
    """The coefficients, the faces' d, the imbalance and the right-hand side."""
    with np.errstate(all="ignore"):
        expected = room.reference.coefficients(*inputs.diagonals)
        faces = room.reference._face_d(*inputs.diagonals)
    got = room.core.coefficients(*inputs.diagonals)
    assert {n: _differ(getattr(expected, n), got[n]) for n in FIVE} == dict.fromkeys(
        FIVE, 0
    )
    core_faces = room.core.face_d(*inputs.diagonals)
    assert [_differ(a, b) for a, b in zip(faces, core_faces, strict=True)] == [0, 0]
    imbalance = room.reference.mass_imbalance(*inputs.stars)
    assert _differ(imbalance, room.core.mass_imbalance(*inputs.stars)) == 0
    f = room.core.right_hand_side(*inputs.stars, *inputs.diagonals)
    assert _differ(_handed(room, inputs), f) == 0


def _assert_correction(room: Room, inputs: Inputs, rng: np.random.Generator) -> None:
    """The correction by a planted p', SOLID cells' values among it."""
    solved = rng.standard_normal(inputs.p.shape)
    expected = _corrected_by(room, inputs, solved)
    got = room.core.apply(*inputs.arguments, solved)
    assert {n: _differ(getattr(expected, n), got[n]) for n in FIELDS} == dict.fromkeys(
        FIELDS, 0
    )


@pytest.mark.parametrize("faces", ["unknowns", "some", "any"])
@pytest.mark.parametrize("spread", list(SPREADS))
@pytest.mark.parametrize("name", NAMES)
def test_the_equation_and_the_correction_are_bit_for_bit(
    name: str, spread: str, faces: str
) -> None:
    """The first kind's blocks of a correction, closed and with outlets (prediction (a))."""
    rng = np.random.default_rng([SEED, NAMES.index(name), len(spread), len(faces)])
    for room in _rooms(name):
        inputs = _planted(room, rng, SPREADS[spread], faces)
        _assert_equation(room, inputs)
        _assert_correction(room, inputs, rng)


def test_the_planted_rooms_are_closed_and_open() -> None:
    """The closed rooms have a pin and the open ones do not, as the core reads them too."""
    for name in NAMES:
        closed, opened = _rooms(name)
        assert closed.reference.needs_pin and not opened.reference.needs_pin
        for room in (closed, opened):
            assert room.core.needs_pin == room.reference.needs_pin
            if room.reference.needs_pin:
                assert room.core.pin_cell == room.reference.pin_cell
    # A pin that is not the first cell, and an outlet on every edge.
    assert _closed("80x30").reference.pin_cell != (0, 0)
    outlets = _open("80x30").reference
    assert all(
        o.any()
        for o in (
            outlets._out_left,
            outlets._out_right,
            outlets._out_bottom,
            outlets._out_top,
        )
    )


def _assert_blocks(five: list[np.ndarray], rng: np.random.Generator) -> None:
    """The operator, the diagonal preconditioner, a sweep and the projection on one system."""
    c = PressureCoefficients(*five)
    shape = c.a_p.shape
    x, r = rng.standard_normal(shape), rng.standard_normal(shape)
    r[rng.random(shape) < 0.1] = -0.0
    assert (
        _differ(pressure.apply_operator(c, x), _cfdcore.apply_operator(*five, x)) == 0
    )
    jacobi = pressure.jacobi_preconditioner(c)(r)
    got = _cfdcore.pressure_precondition(*five, r, PRESSURE_JACOBI, False, LEVELS)
    assert _differ(jacobi, got) == 0
    smoother = rng.random(shape)
    expected = x + smoother * (r - pressure.apply_operator(c, x))
    assert _differ(expected, _cfdcore.smooth(*five, smoother, x, r)) == 0
    active = c.a_p > 0.0
    if active.any():
        expected = x.copy()
        expected[active] -= expected[active].mean()
        assert _differ(expected, _cfdcore.project_mean(x, active)) == 0


def _assert_groups(shape: tuple[int, int], rng: np.random.Generator) -> None:
    """The group sums and the prolongation on fields of one shape, negative zeros among them."""
    r = rng.standard_normal(shape) * 10.0 ** rng.uniform(-8, 8, shape)
    r[rng.random(shape) < 0.5] = -0.0
    sums = multigrid.restrict(r)
    assert _differ(sums, _cfdcore.restrict(r)) == 0
    z = rng.standard_normal(sums.shape)
    expected = rng.standard_normal(shape)
    got = expected.copy()
    multigrid.add_prolonged(expected, z)
    _cfdcore.add_prolonged(got, z)
    assert _differ(expected, got) == 0


def _assert_levels(five: list[np.ndarray]) -> Multigrid:
    """Every level's arrays, its smoother and the coarsest matrix; returns the reference's."""
    c = PressureCoefficients(*five)
    expected = Multigrid(c, pressure.apply_operator)
    core = _cfdcore.Multigrid2(*five, LEVELS)
    levels = core.levels()
    assert len(levels) == len(expected.levels)
    for level, got in zip(expected.levels, levels, strict=True):
        for name in FIVE:
            assert _differ(getattr(level.coefficients, name), got[name]) == 0
        if level.weight == 0.0:
            assert got["smoother"] is None
        else:
            assert _differ(level.smoother, got["smoother"]) == 0
    matrix = _dense_matrix(expected.levels[-1].coefficients)
    assert _differ(matrix, core.coarsest_matrix()) == 0
    # One coarsening of any system, a coarsest one among them.
    coarse, _active = multigrid.coarsen(c, c.a_p > 0.0)
    got = _cfdcore.coarsen(*five)
    assert {n: _differ(getattr(coarse, n), got[n]) for n in FIVE} == dict.fromkeys(
        FIVE, 0
    )
    return expected


@pytest.mark.parametrize("faces", ["unknowns", "some", "any"])
@pytest.mark.parametrize("spread", list(SPREADS))
@pytest.mark.parametrize("name", NAMES)
def test_the_multigrids_blocks_are_bit_for_bit(
    name: str, spread: str, faces: str
) -> None:
    """The operator, the levels and the passes between them (prediction (a))."""
    rng = np.random.default_rng([SEED, NAMES.index(name), len(spread), len(faces), 1])
    for room in _rooms(name):
        _c, five = _system(room, _planted(room, rng, SPREADS[spread], faces))
        _assert_blocks(five, rng)
        _assert_groups(five[0].shape, rng)
        _assert_levels(five)


SMALL_WIDTHS = [*range(1, 20), 26, 33, 34]
SMALL_HEIGHTS = [1, 2, 3, 4, 5, 8, 9, 17]


@pytest.mark.parametrize(
    ("ny", "nx"),
    # One cell alone has no face between two, so no equation.
    [(ny, nx) for nx in SMALL_WIDTHS for ny in SMALL_HEIGHTS if nx * ny > 1],
)
def test_every_block_is_bit_for_bit_at_every_row_length(ny: int, nx: int) -> None:
    """Every number of cells left after the blocks, odd sides, one cell along a side."""
    rng = np.random.default_rng([SEED, nx, ny])
    for opened in (False, True):
        room = _small_room(ny, nx, rng, opened, solid=0.15 if min(nx, ny) > 2 else 0.0)
        for faces in ("unknowns", "some", "any"):
            inputs = _planted(room, rng, 2.0, faces)
            _assert_equation(room, inputs)
            _assert_correction(room, inputs, rng)
            _c, five = _system(room, inputs)
            _assert_blocks(five, rng)
            _assert_groups((ny, nx), rng)
            _assert_levels(five)


# ---- The second kind, to 1e-12 -----------------------------------------------


@pytest.mark.parametrize("name", NAMES)
def test_the_pseudo_inverse_and_a_cycle_are_within_rounding(name: str) -> None:
    """The coarsest pseudo-inverse and a whole V-cycle, closed and with outlets (prediction (b))."""
    rng = np.random.default_rng([SEED, NAMES.index(name), 2])
    for room in _rooms(name):
        for spread in SPREADS.values():
            c, five = _system(room, _planted(room, rng, spread))
            expected = Multigrid(c, pressure.apply_operator)
            core = _cfdcore.Multigrid2(*five, LEVELS)
            inverse = core.coarsest_inverse()
            assert _largest(inverse, expected.coarsest_inverse) <= TOLERANCE
            assert np.array_equal(inverse, inverse.T)
            for _ in range(3):
                r = np.where(c.a_p > 0.0, rng.standard_normal(c.a_p.shape), 0.0)
                z = core.cycle(r)
                assert _largest(z, expected.cycle(r)) <= TOLERANCE
                assert not z[c.a_p <= 0.0].any()
                for project in (False, True):
                    want = pressure.multigrid_preconditioner(c, project)(r)
                    got = _cfdcore.pressure_precondition(
                        *five, r, MULTIGRID, project, LEVELS
                    )
                    assert _largest(got, want) <= TOLERANCE


def _neumann(
    ny: int, nx: int, rng: np.random.Generator, spread: float, solid: float = 0.0
) -> list[np.ndarray]:
    """A closed system of random couplings: singular, with a null vector a component."""
    free = rng.random((ny, nx)) >= solid
    across_x = 10.0 ** rng.uniform(-spread / 2, spread / 2, (ny, nx + 1))
    across_y = 10.0 ** rng.uniform(-spread / 2, spread / 2, (ny + 1, nx))
    across_x[:, [0, -1]] = 0.0
    across_y[[0, -1], :] = 0.0
    across_x[:, 1:-1][~(free[:, 1:] & free[:, :-1])] = 0.0
    across_y[1:-1, :][~(free[1:, :] & free[:-1, :])] = 0.0
    a_e, a_w = across_x[:, 1:].copy(), across_x[:, :-1].copy()
    a_n, a_s = across_y[1:, :].copy(), across_y[:-1, :].copy()
    return [a_e + a_w + a_n + a_s, a_e, a_w, a_n, a_s]


# The shapes of coarsest levels, up to 64 cells or one along a side, and the
# shapes of the levels above them.
COARSEST = [
    *((1, 12), (12, 1), (2, 2), (2, 9), (3, 5), (4, 10), (8, 8), (7, 9)),
    *((10, 13), (16, 16), (33, 33), (3, 40), (40, 3), (30, 80)),
]
# The criterion (docs/reports/phase6_pressure.md, section 4): each pseudo-inverse
# within this many n kappa eps max|X| of the referee's X, and the two within twice
# that of each other.
REFEREE_FACTOR = 4.0


def planted_coarsest(
    shape: tuple[int, int],
) -> Iterator[tuple[tuple[float, float, float], list[np.ndarray]]]:
    """The planted systems of one shape: closed ones, and some with an outlet's term in a corner.

    Yields
    ------
    tuple[tuple[float, float, float], list[np.ndarray]]
        The share of SOLID cells, the couplings' spread in decades and the
        outlet's term, and the five arrays. Every closed one is singular, as is
        every one whose SOLID cells part it.
    """
    rng = np.random.default_rng([SEED, *shape])
    for solid in (0.0, 0.2):
        for spread in (0.0, 0.5, 1.0):
            for outlet in (0.0, 0.3):
                five = _neumann(*shape, rng, spread, solid)
                five[0][0, 0] += outlet * (five[0][0, 0] > 0.0)
                yield (solid, spread, outlet), five


@pytest.mark.parametrize("shape", COARSEST, ids=lambda s: f"{s[0]}x{s[1]}")
def test_the_pseudo_inverse_of_singular_systems(shape: tuple[int, int]) -> None:
    """Closed systems, one null vector a component and one a SOLID cell (prediction (b))."""
    rcond = core_mapping.PINV_RCOND
    for _case, five in planted_coarsest(shape):
        core = _cfdcore.Multigrid2(*five, LEVELS)
        expected = Multigrid(PressureCoefficients(*five), pressure.apply_operator)
        values = np.sort(_cfdcore.symmetric_eigenvalues(core.coarsest_matrix()))
        lapack = np.sort(np.abs(np.linalg.eigvalsh(core.coarsest_matrix())))
        assert np.abs(values - lapack).max() <= TOLERANCE * lapack[-1]
        # The rank is NumPy's, with no value near the cutoff on either side.
        kept = lapack > rcond * lapack[-1]
        assert np.array_equal(values > rcond * values[-1], kept)
        assert (values[~kept] < 0.1 * rcond * values[-1]).all()
        judged = decimal_inverse.referee(core.coarsest_matrix(), rcond)
        assert judged.dropped == np.count_nonzero(~kept)
        own, theirs = core.coarsest_inverse(), expected.coarsest_inverse
        bound = judged.bound(REFEREE_FACTOR)
        assert np.abs(own - judged.inverse).max() <= bound
        assert np.abs(theirs - judged.inverse).max() <= bound
        assert np.abs(own - theirs).max() <= judged.bound(2.0 * REFEREE_FACTOR)


def test_the_pseudo_inverses_cutoff_is_numpys() -> None:
    """Diagonal matrices with a value either side of the cutoff, kept and dropped as NumPy does."""
    rcond = core_mapping.PINV_RCOND
    for factor in (0.0, 1e-3, 0.5, 0.99, 1.01, 2.0, 1e3):
        small = factor * rcond * 7.0
        matrix = np.diag([7.0, small, 2.0])
        expected = np.linalg.pinv(matrix, hermitian=True)
        got = _cfdcore.pseudo_inverse(matrix, rcond)
        assert (got[1, 1] > 0.0) == (expected[1, 1] > 0.0) == (factor > 1.0)
        assert np.allclose(got, expected, rtol=1e-12, atol=0.0)
    # And at another cutoff.
    got = _cfdcore.pseudo_inverse(np.diag([7.0, 1.0, 2.0]), 0.2)
    assert got[1, 1] == 0.0 and got[2, 2] == pytest.approx(0.5)
    assert not _cfdcore.pseudo_inverse(np.zeros((4, 4)), rcond).any()


def test_a_value_that_is_not_finite_makes_a_pseudo_inverse_of_nans() -> None:
    """The core's own rule, so that a diverged field stays one; NumPy's result varies with its LAPACK."""
    for bad in (np.nan, np.inf):
        matrix = np.eye(6) * 2.0
        matrix[0, 0] = bad
        assert np.isnan(_cfdcore.pseudo_inverse(matrix, core_mapping.PINV_RCOND)).all()


# ---- The conjugate gradients -------------------------------------------------


def _assert_solved(room: Room, inputs: Inputs) -> tuple[int, int]:
    """A correction at the reference's count with its fields to 1e-12; returns both counts."""
    expected = room.reference.correct(inputs.prediction, inputs.p)
    got = core_mapping.correction_from(room.core.correct(*inputs.arguments))
    counts = expected.iterations, got.iterations
    if counts[0] == counts[1]:
        assert (got.reached_cap, got.products) == (
            expected.reached_cap,
            expected.products,
        )
        for name in FIELDS:
            assert _largest(getattr(got, name), getattr(expected, name)) <= TOLERANCE
    return counts


@pytest.mark.parametrize("preconditioner", [PRESSURE_JACOBI, MULTIGRID])
@pytest.mark.parametrize("opened", [False, True], ids=["closed", "open"])
@pytest.mark.parametrize("name", NAMES)
def test_a_correction_stops_at_the_references_count(
    name: str, opened: bool, preconditioner: str
) -> None:
    """Counts equal and fields to 1e-12 on every mesh, under both preconditioners (prediction (c))."""
    rng = np.random.default_rng([SEED, NAMES.index(name), int(opened), 3])
    room = (_open if opened else _closed)(name, preconditioner)
    counts = [_assert_solved(room, _planted(room, rng)) for _ in range(2)]
    assert all(a == b for a, b in counts), counts
    assert all(a > 3 for a, _b in counts)


@contextlib.contextmanager
def _core_dots() -> Iterator[None]:
    """The reference's loop with the core's dot product in place of ``np.vdot``."""

    class Proxy:
        vdot = staticmethod(_cfdcore.row_dot)

        def __getattr__(self, name: str) -> object:
            return getattr(np, name)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(pressure, "np", Proxy())
        yield


def _core_preconditioner(
    five: list[np.ndarray], preconditioner: str, project: bool
) -> Callable[[np.ndarray], np.ndarray]:
    """The core's preconditioner, called as the reference calls its own."""
    if preconditioner == PRESSURE_JACOBI:
        return pressure.jacobi_preconditioner(PressureCoefficients(*five))
    core = _cfdcore.Multigrid2(*five, LEVELS)
    active = five[0] > 0.0

    def apply(r: np.ndarray) -> np.ndarray:
        z = core.cycle(r)
        return _cfdcore.project_mean(z, active) if project else z

    return apply


# Tolerances from the configurations' to below rounding, where the recursion
# meets the stop and the true residual does not: restarts, and caps.
LEVELS_OF_RTOL = [1e-2, 1e-4, 1e-8, 1e-12, 1e-14, 1e-15, 1e-16]


@pytest.mark.parametrize("rtol", LEVELS_OF_RTOL)
@pytest.mark.parametrize("preconditioner", [PRESSURE_JACOBI, MULTIGRID])
@pytest.mark.parametrize("opened", [False, True], ids=["closed", "open"])
def test_the_loop_is_the_references_bit_for_bit_given_the_cores_parts(
    opened: bool, preconditioner: str, rtol: float
) -> None:
    """The reference's loop handed the core's dots and cycle: every number the same."""
    rng = np.random.default_rng([SEED, int(opened), LEVELS_OF_RTOL.index(rtol)])
    for shape in ((9, 13), (12, 40), (30, 33)):
        room = _small_room(*shape, rng, opened)
        inputs = _planted(room, rng)
        c, five = _system(room, inputs)
        f = _handed(room, inputs)
        project = not opened
        precondition = _core_preconditioner(five, preconditioner, project)
        for floor, cap in ((0.0, 60), (1e-13, 300), (0.05, 300), (0.0, 1), (0.0, 7)):
            with _core_dots():
                expected = pressure.conjugate_gradient(
                    lambda x, c=c: pressure.apply_operator(c, x),
                    precondition,
                    f,
                    rtol,
                    floor,
                    cap,
                )
            got = _cfdcore.pressure_cg(
                *five, f, rtol, floor, cap, preconditioner, project, LEVELS
            )
            assert _differ(expected.x, got["x"]) == 0
            assert (
                expected.iterations,
                expected.reached_cap,
                expected.residual_norm,
                expected.products,
            ) == (
                got["iterations"],
                got["reached_cap"],
                got["residual_norm"],
                got["products"],
            )


@pytest.mark.parametrize("preconditioner", [PRESSURE_JACOBI, MULTIGRID])
def test_the_stop_is_read_at_its_boundary(preconditioner: str) -> None:
    """A floor at each iteration's own residual norm, and one place either side of it.

    The norm is the square root of the core's dot product of the recursion's
    residual, so the stop holds at it exactly (it is inclusive) and a norm in
    another order of summation, a place away, would stop at another count.
    """
    rng = np.random.default_rng([SEED, 10, len(preconditioner)])
    for opened in (False, True):
        room = _small_room(14, 37, rng, opened)
        inputs = _planted(room, rng)
        c, five = _system(room, inputs)
        f = _handed(room, inputs)
        precondition = _core_preconditioner(five, preconditioner, not opened)
        norms: list[float] = []

        class Norms:
            """Keeps every norm the loop takes; its dot product is the core's."""

            vdot = staticmethod(_cfdcore.row_dot)

            def sqrt(self, x: float, kept: list[float] = norms) -> float:
                kept.append(float(np.sqrt(x)))
                return kept[-1]

            def __getattr__(self, name: str) -> object:
                return getattr(np, name)

        def solve(
            floor: float,
            names: object,
            c: PressureCoefficients = c,
            f: np.ndarray = f,
            precondition: Callable[[np.ndarray], np.ndarray] = precondition,
        ) -> pressure.ConjugateGradientResult:
            with pytest.MonkeyPatch.context() as patch:
                patch.setattr(pressure, "np", names)
                return pressure.conjugate_gradient(
                    lambda x: pressure.apply_operator(c, x),
                    precondition,
                    f,
                    0.0,
                    floor,
                    25,
                )

        solve(0.0, Norms())
        # The first norm is the right-hand side's; the rest are the iterations'.
        read = norms[1:21]
        stops = set()
        for norm in read:
            for floor in (np.nextafter(norm, 0.0), norm, np.nextafter(norm, np.inf)):
                expected = solve(float(floor), Norms())
                got = _cfdcore.pressure_cg(
                    *five, f, 0.0, float(floor), 25, preconditioner, not opened, LEVELS
                )
                assert _differ(expected.x, got["x"]) == 0
                counts = (expected.iterations, expected.products)
                assert counts == (got["iterations"], got["products"])
                stops.add(expected.iterations)
        assert len(stops) > 10


def test_the_loops_cases_reach_a_restart_the_floor_and_the_cap() -> None:
    """The cases above restart, stop on the floor and stop at the cap, so they can tell."""
    rng = np.random.default_rng([SEED, 0, 5])
    room = _small_room(12, 40, rng, opened=False)
    inputs = _planted(room, rng)
    _c, five = _system(room, inputs)
    f = _handed(room, inputs)
    solve = functools.partial(_cfdcore.pressure_cg, *five, f)
    tight = solve(1e-16, 0.0, 300, MULTIGRID, True, LEVELS)
    # Products beyond one an iteration and the exit check: confirmations that failed.
    assert tight["products"] > tight["iterations"] + 1
    assert tight["reached_cap"]
    floor = solve(1e-16, 0.05, 300, MULTIGRID, True, LEVELS)
    assert not floor["reached_cap"] and floor["residual_norm"] <= 0.05
    assert floor["products"] == floor["iterations"] + 1


@pytest.mark.parametrize("preconditioner", [PRESSURE_JACOBI, MULTIGRID])
@pytest.mark.parametrize("opened", [False, True], ids=["closed", "open"])
def test_corrections_of_small_rooms(opened: bool, preconditioner: str) -> None:
    """Rooms of many shapes and SOLID cells: counts equal at the configurations' tolerances."""
    rng = np.random.default_rng([SEED, int(opened), len(preconditioner)])
    counts = []
    for ny, nx in ((2, 2), (3, 7), (9, 9), (5, 33), (17, 12), (26, 31), (3, 64)):
        for rtol in (1e-4, 1e-8):
            room = _small_room(ny, nx, rng, opened, preconditioner, rtol=rtol)
            counts.append(_assert_solved(room, _planted(room, rng)))
    assert all(a == b for a, b in counts), counts


def test_the_cap_is_reported() -> None:
    """A correction the cap ends returns its fields, its counts and the flag."""
    rng = np.random.default_rng([SEED, 6])
    for preconditioner in (PRESSURE_JACOBI, MULTIGRID):
        room = _small_room(12, 31, rng, False, preconditioner, rtol=1e-12, max_iter=3)
        inputs = _planted(room, rng)
        assert _assert_solved(room, inputs) == (3, 3)
        got = room.core.correct(*inputs.arguments)
        assert got["reached_cap"] and got["products"] == 4
        assert got["residual_norm"] > 0.0


@pytest.mark.parametrize("preconditioner", [PRESSURE_JACOBI, MULTIGRID])
def test_a_zero_right_hand_side_takes_no_iteration(preconditioner: str) -> None:
    """Predicted velocities at rest: p' is zero, and p loses the reference cell's value alone."""
    rng = np.random.default_rng([SEED, 7])
    for opened in (False, True):
        room = _small_room(9, 14, rng, opened, preconditioner)
        inputs = _planted(room, rng)
        inputs.u_star[...] = 0.0
        inputs.v_star[...] = 0.0
        expected = room.reference.correct(inputs.prediction, inputs.p)
        got = room.core.correct(*inputs.arguments)
        for name in FIELDS:
            assert _differ(getattr(expected, name), got[name]) == 0
        assert (got["iterations"], got["products"]) == (0, 0)
        assert not got["reached_cap"] and got["residual_norm"] == 0.0


def test_one_corrector_through_many_calls_is_each_calls_own() -> None:
    """A correction leaves nothing the next one reads."""
    rng = np.random.default_rng([SEED, 8])
    room = _small_room(17, 26, rng, opened=False)
    calls = [_planted(room, rng) for _ in range(4)]
    fresh = [
        _room(
            room.mesh,
            room.config,
            PlantedBoundary(_no_outlets(room.mesh)),
        ).core.correct(*inputs.arguments)
        for inputs in calls
    ]
    for inputs, expected in zip((*calls, *calls), (*fresh, *fresh), strict=True):
        got = room.core.correct(*inputs.arguments)
        assert [_differ(expected[n], got[n]) for n in FIELDS] == [0] * len(FIELDS)
        # The other calls between two corrections change nothing either.
        room.core.coefficients(*calls[0].diagonals)
        room.core.mass_imbalance(*calls[0].stars)


def test_a_nan_lands_where_the_references_does() -> None:
    """A NaN velocity spreads to the fields the reference's spreads to, at the cap."""
    rng = np.random.default_rng([SEED, 9])
    for preconditioner in (PRESSURE_JACOBI, MULTIGRID):
        for opened in (False, True):
            room = _small_room(9, 14, rng, opened, preconditioner, max_iter=4)
            inputs = _planted(room, rng)
            j, i = np.argwhere(inputs.a_p_u > 0.0)[0]
            inputs.u_star[j, i] = np.nan
            with np.errstate(all="ignore"):
                expected = room.reference.correct(inputs.prediction, inputs.p)
            got = room.core.correct(*inputs.arguments)
            assert (got["iterations"], got["reached_cap"]) == (4, True)
            assert expected.iterations == 4 and expected.reached_cap
            for name in FIELDS:
                want = getattr(expected, name)
                assert np.array_equal(np.isnan(want), np.isnan(got[name]))
                finite = ~np.isnan(want)
                assert np.array_equal(want[finite], got[name][finite])


# ---- The refusals ------------------------------------------------------------


@pytest.mark.parametrize(
    ("rtol", "floor", "max_iter"),
    [
        *(
            (bad, 0.0, 10)
            for bad in (-1e-9, 1.0, 2, np.nan, True, "x", None, np.float32(0.1))
        ),
        *((1e-4, bad, 10) for bad in (-1e-30, np.inf, np.nan, False, "x", None)),
        *((1e-4, 0.0, bad) for bad in (0, -3, 1.0, True, "x", None, np.int64(5))),
    ],
)
def test_the_solves_numbers_are_refused_as_the_reference_refuses_them(
    rtol: object, floor: object, max_iter: object
) -> None:
    """The types and messages of ``_check_solve_arguments``, at the solve and the corrector."""
    five = _neumann(3, 4, np.random.default_rng(1), 0.0)
    f = np.zeros((3, 4))
    with pytest.raises(ValueError) as expected:
        pressure.conjugate_gradient(lambda x: x, lambda r: r, f, rtol, floor, max_iter)
    with pytest.raises(ValueError) as got:
        _cfdcore.pressure_cg(
            *five, f, rtol, floor, max_iter, PRESSURE_JACOBI, False, LEVELS
        )
    assert str(got.value) == str(expected.value)
    room = _closed("80x30")
    with pytest.raises(ValueError) as built:
        _corrector(room.mesh, rtol=rtol, floor=floor, max_iter=max_iter)
    assert str(built.value) == str(expected.value)


def test_a_floor_too_large_for_a_float_overflows_as_the_references_does() -> None:
    """``math.isfinite`` of such an int raises OverflowError, and so does the core."""
    five = _neumann(3, 4, np.random.default_rng(1), 0.0)
    f = np.zeros((3, 4))
    for error in (
        lambda: pressure.conjugate_gradient(
            lambda x: x, lambda r: r, f, 0.1, 10**400, 5
        ),
        lambda: _cfdcore.pressure_cg(
            *five, f, 0.1, 10**400, 5, PRESSURE_JACOBI, False, LEVELS
        ),
    ):
        with pytest.raises(OverflowError, match="int too large to convert to float"):
            error()
    with pytest.raises(ValueError, match="rtol must be a number in"):
        _cfdcore.pressure_cg(*five, f, 10**400, 0.0, 5, PRESSURE_JACOBI, False, LEVELS)


def test_the_solves_numbers_that_are_accepted() -> None:
    """Integers where the reference takes them, NumPy's float, and a cap beyond 64 bits."""
    five = _neumann(3, 4, np.random.default_rng(1), 0.0)
    f = np.zeros((3, 4))
    for rtol, floor, cap in (
        (0, 0, 1),
        (np.float64(0.5), 10**20, 10**30),
        (0.999, 1e300, 2),
    ):
        got = _cfdcore.pressure_cg(
            *five, f, rtol, floor, cap, PRESSURE_JACOBI, False, LEVELS
        )
        assert got["iterations"] == 0


def _corrector(
    mesh: object,
    outlets: object = None,
    rtol: object = 1e-4,
    floor: object = 0.0,
    max_iter: object = 100,
    **keys: object,
) -> "_cfdcore.PressureCorrector2":
    """A core corrector with any of its arguments replaced."""
    none = _no_outlets(mesh)
    arguments = {
        "dx_cell": mesh.dx_cell,
        "dy_cell": mesh.dy_cell,
        "outlets": [none[e] for e in EDGES] if outlets is None else outlets,
        "fluid": FLUID,
        "rho": 1.2,
        "alpha": 0.3,
        "rtol": rtol,
        "floor": floor,
        "max_iter": max_iter,
        "preconditioner": MULTIGRID,
        "levels": LEVELS,
    }
    grid = _cfdcore.Grid2(mesh.x, mesh.xc, mesh.y, mesh.yc, mesh.cell_type, SOLID)
    return _cfdcore.PressureCorrector2(grid, **{**arguments, **keys})


def _walled(opened: dict[str, np.ndarray] | None) -> tuple[object, PlantedBoundary]:
    """A room of two halves with a wall between, and its boundary."""
    mesh = _named_mesh(CAVITY)
    cell_type = mesh.cell_type.copy()
    cell_type[:, 20] = SOLID
    # A sealed single cell, which has no equation and is no component.
    cell_type[4:7, 30:33] = SOLID
    cell_type[5, 31] = FLUID
    axes = ("x", "xc", "y", "yc", "dx_cell", "dy_cell")
    walled = SimpleNamespace(
        **{name: getattr(mesh, name) for name in axes}, cell_type=cell_type
    )
    return walled, PlantedBoundary(_no_outlets(mesh) if opened is None else opened)


@pytest.mark.parametrize("case", ["closed", "stranded-left", "stranded-both"])
def test_the_components_are_refused_as_the_reference_refuses_them(case: str) -> None:
    """A closed domain of two components, and an open one with one that reaches no outlet."""
    opened = None
    if case != "closed":
        opened = _no_outlets(_named_mesh(CAVITY))
        if case == "stranded-left":
            opened["right"][3] = True
        else:
            # Beside a SOLID cell: its face borrows nothing, so it reaches nothing.
            opened["top"][20] = True
    mesh, boundary = _walled(opened)
    with pytest.raises(ValueError) as expected:
        PressureCorrector(mesh, _solver_config(CAVITY), boundary)
    with pytest.raises(ValueError) as got:
        _corrector(
            mesh,
            [boundary.pressure_outlets()[e].is_outlet for e in EDGES],
        )
    assert str(got.value) == str(expected.value)
    if case == "stranded-left":
        # With one on each side both accept it.
        opened["left"][39] = True
        _corrector(mesh, [opened[e] for e in EDGES])
        PressureCorrector(mesh, _solver_config(CAVITY), PlantedBoundary(opened))


def test_components_on_random_grids_are_the_references() -> None:
    """Random SOLID cells and outlets: the core refuses what the reference refuses, as it does."""
    rng = np.random.default_rng([SEED, 11])
    refused = accepted = 0
    for trial in range(400):
        ny, nx = (int(n) for n in rng.integers(1, 9, 2))
        cell_type = np.where(rng.random((ny, nx)) < 0.35, SOLID, FLUID).astype(np.int32)
        x, y = np.arange(nx + 1.0), np.arange(ny + 1.0)
        mesh = SimpleNamespace(
            x=x,
            y=y,
            xc=x[:-1] + 0.5,
            yc=y[:-1] + 0.5,
            dx_cell=np.diff(x),
            dy_cell=np.diff(y),
            cell_type=cell_type,
        )
        share = 0.0 if trial % 3 == 0 else 0.5
        outlets = {e: rng.random(n) < share for e, n in _edge_lengths(mesh).items()}
        messages = []
        for build in (
            lambda m=mesh, o=outlets: PressureCorrector(
                m, _solver_config(CAVITY), PlantedBoundary(o)
            ),
            lambda m=mesh, o=outlets: _corrector(m, [o[e] for e in EDGES]),
        ):
            try:
                build()
                messages.append(None)
            except ValueError as error:
                messages.append(str(error))
        assert messages[0] == messages[1]
        refused += messages[0] is not None
        accepted += messages[0] is None
    assert min(refused, accepted) > 50


def test_shapes_are_refused_as_the_reference_refuses_them() -> None:
    """The velocities, the diagonals and p, with the reference's types and messages."""
    rng = np.random.default_rng([SEED, 12])
    room = _small_room(6, 9, rng, opened=False)
    good = _planted(room, rng)
    for field in ("u_star", "v_star", "p", "a_p_u", "a_p_v"):
        for wrong in (np.zeros((6, 8)), np.zeros(5), np.zeros((2, 3, 4))):
            inputs = dataclasses.replace(good, **{field: wrong})
            with pytest.raises(ValueError) as expected:
                room.reference.correct(inputs.prediction, inputs.p)
            with pytest.raises(ValueError) as got:
                room.core.correct(*inputs.arguments)
            assert str(got.value) == str(expected.value)
            with pytest.raises(ValueError):
                room.core.apply(*inputs.arguments, good.p)
            if field in ("u_star", "v_star"):
                with pytest.raises(ValueError) as expected:
                    room.reference.mass_imbalance(*inputs.stars)
                with pytest.raises(ValueError) as got:
                    room.core.mass_imbalance(*inputs.stars)
                assert str(got.value) == str(expected.value)
                with pytest.raises(ValueError):
                    room.core.right_hand_side(*inputs.stars, *inputs.diagonals)
            if field in ("a_p_u", "a_p_v"):
                with pytest.raises(ValueError) as expected:
                    room.reference.coefficients(*inputs.diagonals)
                for call in (room.core.coefficients, room.core.face_d):
                    with pytest.raises(ValueError) as got:
                        call(*inputs.diagonals)
                    assert str(got.value) == str(expected.value)
                with pytest.raises(ValueError):
                    room.core.right_hand_side(*inputs.stars, *inputs.diagonals)
    with pytest.raises(ValueError, match="expected p_prime of shape"):
        room.core.apply(*good.arguments, np.zeros((6, 8)))


def test_arrays_the_core_does_not_read_are_refused() -> None:
    """Another dtype, layout or alignment is a TypeError at every entry."""
    rng = np.random.default_rng([SEED, 13])
    room = _small_room(6, 9, rng, opened=False)
    good = _planted(room, rng)
    five = _neumann(6, 9, rng, 0.0)
    field = np.zeros((6, 9))
    spoilers = (
        _fortran,
        _strided,
        _unaligned_like,
        lambda a: a.astype(np.float32),
        lambda a: a.tolist(),
    )
    entries: list[tuple[Callable[..., object], tuple[object, ...]]] = [
        (room.core.correct, good.arguments),
        (room.core.apply, (*good.arguments, field)),
        (room.core.coefficients, good.diagonals),
        (room.core.face_d, good.diagonals),
        (room.core.mass_imbalance, good.stars),
        (room.core.right_hand_side, (*good.stars, *good.diagonals)),
        (_cfdcore.apply_operator, (*five, field)),
        (_cfdcore.smooth, (*five, field, field, field)),
        (_cfdcore.coarsen, tuple(five)),
        (_cfdcore.restrict, (field,)),
        (_cfdcore.add_prolonged, (field.copy(), np.zeros((3, 5)))),
        (_cfdcore.pseudo_inverse, (np.eye(4), 1e-15)),
        (_cfdcore.symmetric_eigenvalues, (np.eye(4),)),
        (_cfdcore.Multigrid2, (*five, LEVELS)),
        (
            _cfdcore.pressure_cg,
            (*five, field, 0.1, 0.0, 3, PRESSURE_JACOBI, False, LEVELS),
        ),
        (
            _cfdcore.pressure_precondition,
            (*five, field, PRESSURE_JACOBI, False, LEVELS),
        ),
    ]
    for call, arguments in entries:
        call(*arguments)
        for at, argument in enumerate(arguments):
            if not isinstance(argument, np.ndarray):
                continue
            for spoil in spoilers:
                spoilt = [*arguments]
                spoilt[at] = spoil(argument)
                with pytest.raises(TypeError):
                    call(*spoilt)
    with pytest.raises(TypeError):
        _cfdcore.project_mean(_unaligned_like(field), field > 0)
    with pytest.raises(TypeError):
        _cfdcore.project_mean(field, np.zeros((6, 8), dtype=bool))
    with pytest.raises(TypeError, match="not square"):
        _cfdcore.pseudo_inverse(np.zeros((3, 4)), 1e-15)
    x = np.zeros(8)
    with pytest.raises(TypeError, match="shares memory"):
        _cfdcore.add_prolonged(x.reshape(1, 8), x[:4].reshape(1, 4))


def test_fields_of_another_shape_are_refused() -> None:
    """A field that is not the system's shape is a ValueError naming both shapes."""
    five = _neumann(6, 9, np.random.default_rng(1), 0.0)
    field, wrong = np.zeros((6, 9)), np.zeros((9, 6))
    for call in (
        lambda: _cfdcore.apply_operator(*five, wrong),
        lambda: _cfdcore.smooth(*five, wrong, field, field),
        lambda: _cfdcore.smooth(*five, field, wrong, field),
        lambda: _cfdcore.smooth(*five, field, field, wrong),
        lambda: _cfdcore.Multigrid2(*five, LEVELS).cycle(wrong),
        lambda: _cfdcore.pressure_cg(
            *five, wrong, 0.1, 0.0, 3, PRESSURE_JACOBI, False, LEVELS
        ),
        lambda: _cfdcore.pressure_precondition(
            *five, wrong, PRESSURE_JACOBI, False, LEVELS
        ),
    ):
        with pytest.raises(ValueError, match=r"of shape \(6, 9\), got \(9, 6\)"):
            call()
    with pytest.raises(TypeError, match="z is not"):
        _cfdcore.add_prolonged(np.zeros((6, 9)), np.zeros((3, 4)))


def test_the_multigrids_refusals_are_the_references() -> None:
    """Five arrays of one two-dimensional shape, and a weakly dominant finest system."""
    five = _neumann(9, 13, np.random.default_rng(2), 0.5)
    spoilt = []
    for at in range(5):
        for wrong in (np.zeros((9, 12)), np.zeros(117), np.zeros((9, 13, 1))):
            spoilt.append([wrong if k == at else a for k, a in enumerate(five)])
    spoilt.append([np.zeros(4)] * 5)
    for arrays in spoilt:
        with pytest.raises(ValueError) as expected:
            Multigrid(PressureCoefficients(*arrays), pressure.apply_operator)
        with pytest.raises(ValueError) as got:
            _cfdcore.Multigrid2(*arrays, LEVELS)
        assert str(got.value) == str(expected.value)
    for j, i in ((0, 0), (8, 12), (4, 6), (0, 12)):
        for shortfall, refuses in ((1e-9, True), (3e-12, True), (1e-13, False)):
            weak = [a.copy() for a in five]
            weak[0][j, i] *= 1.0 - shortfall
            reference = functools.partial(
                Multigrid, PressureCoefficients(*weak), pressure.apply_operator
            )
            if not refuses:
                reference()
                _cfdcore.Multigrid2(*weak, LEVELS)
                continue
            with pytest.raises(ValueError) as expected:
                reference()
            with pytest.raises(ValueError) as got:
                _cfdcore.Multigrid2(*weak, LEVELS)
            assert str(got.value) == str(expected.value)
            with pytest.raises(ValueError) as got:
                _cfdcore.pressure_cg(
                    *weak, np.zeros((9, 13)), 0.1, 0.0, 3, MULTIGRID, False, LEVELS
                )
            assert str(got.value) == str(expected.value)
            # The diagonal preconditioner does not ask for it, nor does the reference's.
            _cfdcore.pressure_cg(
                *weak, np.zeros((9, 13)), 0.1, 0.0, 3, PRESSURE_JACOBI, False, LEVELS
            )


def test_a_division_by_zero_is_the_references_error() -> None:
    """A right-hand side in the operator's null space, where the reference's division raises."""
    five = _neumann(1, 2, np.random.default_rng(1), 0.0)
    f = np.ones((1, 2))
    c = PressureCoefficients(*five)
    with pytest.raises(ZeroDivisionError):
        pressure.conjugate_gradient(
            lambda x: pressure.apply_operator(c, x),
            pressure.jacobi_preconditioner(c),
            f,
            0.1,
            0.0,
            3,
        )
    with pytest.raises(_cfdcore.PressureBreakdown) as got:
        _cfdcore.pressure_cg(*five, f, 0.1, 0.0, 3, PRESSURE_JACOBI, False, LEVELS)
    assert got.value.reason == "division_by_zero" and got.value.iterations == 0
    assert isinstance(core_mapping.breakdown_error(got.value), ZeroDivisionError)
    other = SimpleNamespace(reason="not_converged", iterations=0)
    assert isinstance(core_mapping.breakdown_error(other), np.linalg.LinAlgError)


def test_the_constructors_refusals() -> None:
    """What the core's constructor refuses that the reference's configuration has refused."""
    mesh = _closed("80x30").mesh
    none = _no_outlets(mesh)
    good = [none[e] for e in EDGES]
    for at in range(4):
        for wrong in (
            good[at][:-1],
            good[at].astype(np.int8),
            np.zeros((2, 3), dtype=bool),
            None,
        ):
            spoilt = [wrong if k == at else o for k, o in enumerate(good)]
            with pytest.raises(TypeError, match=f"{EDGES[at]} outlets"):
                _corrector(mesh, spoilt)
    with pytest.raises(TypeError, match="left, right, bottom and top"):
        _corrector(mesh, good[:3])
    for key in ("rho", "alpha"):
        for bad in (0.0, -1.0, np.nan, np.inf):
            with pytest.raises(ValueError, match=f"{key} is not finite and positive"):
                _corrector(mesh, **{key: bad})
        for bad in (True, np.bool_(True), "x", None):
            with pytest.raises(TypeError):
                _corrector(mesh, **{key: bad})
    for key in ("dx_cell", "dy_cell"):
        widths = getattr(mesh, key)
        with pytest.raises(ValueError, match="one per cell"):
            _corrector(mesh, **{key: widths[:-1].copy()})
        spoilt = widths.copy()
        spoilt[-1] = -1.0
        with pytest.raises(ValueError, match="cell width is not finite"):
            _corrector(mesh, **{key: spoilt})
        with pytest.raises(TypeError):
            _corrector(mesh, **{key: _unaligned_like(widths)})
    for bad in ("diagonal", None, 3):
        with pytest.raises(ValueError, match="'jacobi' or 'multigrid'"):
            _corrector(mesh, preconditioner=bad)
    for bad in (True, 0.0, "x"):
        with pytest.raises(TypeError):
            _corrector(mesh, fluid=bad)


@pytest.mark.parametrize(
    ("at", "bad", "error"),
    [
        (0, 0, ValueError),
        (0, 1.5, TypeError),
        (0, True, TypeError),
        (1, 0, ValueError),
        (1, 1.0, TypeError),
        (2, 0.0, ValueError),
        (2, np.inf, ValueError),
        (2, np.nan, ValueError),
        (2, True, TypeError),
        (3, -1e-12, ValueError),
        (3, np.inf, ValueError),
        (3, "x", TypeError),
        (4, -1e-15, ValueError),
        (4, np.nan, ValueError),
        (4, None, TypeError),
    ],
)
def test_the_multigrids_keys_are_checked(
    at: int, bad: object, error: type[Exception]
) -> None:
    """Each key out of its range or of another type, wherever the keys are read."""
    five = _neumann(5, 5, np.random.default_rng(1), 0.0)
    levels = tuple(bad if k == at else key for k, key in enumerate(LEVELS))
    field = np.zeros((5, 5))
    for call in (
        lambda: _cfdcore.Multigrid2(*five, levels),
        lambda: _cfdcore.pressure_cg(
            *five, field, 0.1, 0.0, 3, MULTIGRID, False, levels
        ),
        lambda: _cfdcore.pressure_precondition(*five, field, MULTIGRID, False, levels),
        lambda: _corrector(_closed("80x30").mesh, levels=levels),
    ):
        with pytest.raises(error):
            call()
    with pytest.raises(TypeError, match="coarsest_cells, sweeps"):
        _cfdcore.Multigrid2(*five, LEVELS[:4])


def test_the_mappings_keys_are_the_references() -> None:
    """The keys the core is handed are the reference's constants and NumPy's default cutoff."""
    assert LEVELS[:4] == (
        multigrid.COARSEST_CELLS,
        multigrid.SMOOTHING_SWEEPS,
        multigrid.SMOOTHER_FACTOR / 2.0,
        multigrid.DOMINANCE_TOLERANCE,
    )
    # pinv's default: a value at 1.01e-15 of the largest is kept, at 0.99e-15 dropped.
    for factor, kept in ((1.01, True), (0.99, False)):
        small = factor * LEVELS[4]
        inverse = np.linalg.pinv(np.diag([1.0, small]), hermitian=True)
        assert (inverse[1, 1] > 0.0) == kept
