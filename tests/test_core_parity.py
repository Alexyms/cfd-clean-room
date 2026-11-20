"""The compiled core's blocks against the NumPy reference (REQ-N03 level 1, ADR-017 D).

Each block of ``_cfdcore`` is called on the same inputs as the reference and
compared bit for bit where the core copies the reference's order of
operations, as ``reduce.pairwise_sum`` copies ``np.sum``'s. The module is
skipped with its reason when ``_cfdcore`` cannot be imported (the Windows
environment, or a Linux one where ``pip install --no-build-isolation ./core``
has not been run), so the rest of the suite runs as before. With
``CFD_CORE_REQUIRED=1`` in the environment, as CI's core job sets it, a module
that cannot be imported is an error instead of a skip.
"""

import contextlib
import copy
import dataclasses
import functools
import os
from collections.abc import Callable, Iterator
from typing import NamedTuple, Self

import numpy as np
import pytest
import yaml

import src.scalar_scheme as scalar_scheme
from src.boundary_registry import BoundaryRegistry
from src.boundary_vents import vent_face_masks, vent_runs
from src.config import SimConfig
from src.mesh import SOLID, Mesh
from src.momentum import _lagrange_weights, quick_face_values
from src.scalar_scheme import (
    ImplicitResult,
    advective_flux,
    implicit_step,
    limited_face_values,
    mesh_axes,
)

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

pytestmark = [pytest.mark.parity, pytest.mark.integration]

SEED = 86
# Prediction (b)'s lengths: every length to 1,000, and the 80x30, 160x60 and
# 320x120 grids' cell counts.
LENGTHS = [*range(1001), 80 * 30, 160 * 60, 320 * 120]
SHAPES = [(30, 80), (60, 160), (120, 320)]
# Ten to a power in [-DECADES, DECADES] times a standard normal: both signs and
# many magnitudes, so the order of the additions shows in the last bits.
DECADES = 6.0
# Longer than NumPy's buffer (np.getbufsize(), 8,192 terms), which is where an
# unaligned array's sum is taken in chunks and its order parts from an aligned one's.
UNALIGNED_LENGTH = 20_001


def _values(rng: np.random.Generator, shape: int | tuple[int, int]) -> np.ndarray:
    """Random float64 values of both signs over twelve decades."""
    return rng.standard_normal(shape) * 10.0 ** rng.uniform(-DECADES, DECADES, shape)


def _bits(x: float) -> int:
    """Return the 64-bit pattern of x, so that 0.0 and -0.0 differ."""
    return int(np.array([x], dtype=np.float64).view(np.uint64)[0])


def test_pairwise_sum_equals_np_sum_on_every_length() -> None:
    """pairwise_sum is np.sum bit for bit on contiguous 1-D arrays (prediction (b))."""
    rng = np.random.default_rng(SEED)
    differ = []
    order_shows = 0
    for n in LENGTHS:
        values = _values(rng, n)
        if n and np.cumsum(values)[-1] != np.sum(values):
            order_shows += 1
        if _bits(_cfdcore.pairwise_sum(values)) != _bits(np.sum(values)):
            differ.append(n)
    assert differ == []
    # The comparison is sharp only if the order changes the result: a
    # left-to-right sum differs from np.sum on most of these arrays.
    assert order_shows > len(LENGTHS) // 2


def test_pairwise_sum_equals_np_sum_on_grid_shaped_arrays() -> None:
    """A C-contiguous 2-D array's np.sum equals pairwise_sum of its 1-D view."""
    rng = np.random.default_rng(SEED + 1)
    for shape in SHAPES:
        field = _values(rng, shape)
        flat = field.reshape(-1)
        assert np.shares_memory(flat, field)
        assert _bits(_cfdcore.pairwise_sum(flat)) == _bits(np.sum(field)), shape


def test_pairwise_sum_of_negative_zeros_matches_np_sum() -> None:
    """A sum of -0.0 terms is +0.0 in both, as np.sum starts from its identity."""
    for n in (1, 7, 8, 200):
        zeros = np.full(n, -0.0)
        assert _bits(_cfdcore.pairwise_sum(zeros)) == _bits(np.sum(zeros)) == 0


def test_pairwise_sum_accepts_a_read_only_array() -> None:
    """The block reads its input only, so a read-only array is accepted."""
    values = np.arange(10.0)
    values.flags.writeable = False
    assert _cfdcore.pairwise_sum(values) == 45.0


def _unaligned(rng: np.random.Generator, n: int) -> np.ndarray:
    """Return n random float64 values whose data pointer is one byte past a multiple of eight."""
    raw = np.zeros(8 * n + 16, dtype=np.uint8)
    start = (1 - raw.ctypes.data) % 8
    values = raw[start : start + 8 * n].view(np.float64)
    values[:] = _values(rng, n)
    assert not values.flags.aligned
    return values


def test_pairwise_sum_refuses_an_unaligned_array() -> None:
    """An unaligned array is a TypeError, since np.sum sums it in another order."""
    rng = np.random.default_rng(SEED + 2)
    order_changes = 0
    for _ in range(10):
        values = _unaligned(rng, UNALIGNED_LENGTH)
        with pytest.raises(TypeError, match="aligned"):
            _cfdcore.pairwise_sum(values)
        # The same terms in aligned memory are summed, and bit for bit as np.sum sums them.
        aligned = values.copy()
        assert aligned.flags.aligned
        assert _bits(_cfdcore.pairwise_sum(aligned)) == _bits(np.sum(aligned))
        order_changes += _bits(np.sum(values)) != _bits(np.sum(aligned))
    # The refusal is warranted only if np.sum's own value moves with the alignment.
    assert order_changes > 0


@pytest.mark.parametrize(
    "values",
    [
        pytest.param(np.arange(10, dtype=np.float32), id="float32"),
        pytest.param(np.arange(10, dtype=np.int64), id="int64"),
        pytest.param(np.arange(20.0)[::2], id="non-contiguous"),
        pytest.param(np.zeros((3, 4)), id="two-dimensional"),
        pytest.param(np.zeros(()), id="zero-dimensional"),
        pytest.param([1.0, 2.0], id="list"),
    ],
)
def test_pairwise_sum_refuses_other_arrays(values: object) -> None:
    """A wrong dtype, layout or dimension is a TypeError, not a silent copy."""
    with pytest.raises(TypeError):
        _cfdcore.pairwise_sum(values)


# The meshes of prompt 87's parity tests, each (config file, nx, ny, x and y
# stretch ratios, obstacles kept). The three product grids are built as the
# default file builds them: uniform, with the tools' SOLID cells. A stretched
# 160x60 and an obstacle-free uniform grid add the non-uniform weights and a
# field with no SOLID cell.
DEFAULT = "configs/clean_room_default.yaml"
VENTS = "configs/clean_room_vents.yaml"
MESHES = {
    "80x30": (DEFAULT, 80, 30, 1.0, 1.0, True),
    "160x60": (DEFAULT, 160, 60, 1.0, 1.0, True),
    "320x120": (DEFAULT, 320, 120, 1.0, 1.0, True),
    "160x60-stretched": (DEFAULT, 160, 60, 1.02, 1.03, True),
    "40x15-uniform-open": (DEFAULT, 40, 15, 1.0, 1.0, False),
}
# The product grids with the vents file's equipment vents (ADR-014 J); the
# default file has none.
VENT_MESHES = {
    "80x30-vents": (VENTS, 80, 30, 1.0, 1.0, True),
    "160x60-vents": (VENTS, 160, 60, 1.0, 1.0, True),
    "320x120-vents": (VENTS, 320, 120, 1.0, 1.0, True),
}


@functools.cache
def _config(name: str) -> SimConfig:
    """The configuration of one of MESHES or VENT_MESHES."""
    path, nx, ny, sx, sy, obstacles = {**MESHES, **VENT_MESHES}[name]
    with open(path, encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    raw = copy.deepcopy(raw)
    raw["domain"]["nx"] = nx
    raw["domain"]["ny"] = ny
    raw["mesh"]["x"] = {"stretch_ratio": sx}
    raw["mesh"]["y"] = {"stretch_ratio": sy}
    if not obstacles:
        raw["obstacles"] = []
    return SimConfig.from_dict(raw, "configs")


@functools.cache
def _mesh(name: str) -> Mesh:
    """The mesh of one of MESHES or VENT_MESHES."""
    return Mesh(_config(name))


def _grid(mesh: Mesh) -> "_cfdcore.Grid2":
    """The core's grid of a mesh, built from the arrays the mesh holds."""
    return _cfdcore.Grid2(mesh.x, mesh.xc, mesh.y, mesh.yc, mesh.cell_type, SOLID)


def _same_bits(a: np.ndarray, b: np.ndarray) -> bool:
    """Whether two float64 arrays have one shape and equal 64-bit patterns."""
    a = np.ascontiguousarray(a, dtype=np.float64)
    b = np.ascontiguousarray(b, dtype=np.float64)
    return a.shape == b.shape and bool(
        np.array_equal(a.view(np.uint64), b.view(np.uint64))
    )


@pytest.mark.parametrize("name", list(MESHES))
def test_grid_nodes_and_faces_are_mesh_axes(name: str) -> None:
    """Grid2's nodes and interior faces per direction are mesh_axes' bit for bit."""
    mesh = _mesh(name)
    grid = _grid(mesh)
    assert grid.shape == mesh.cell_type.shape
    for d, axis in enumerate(mesh_axes(mesh)):
        assert _same_bits(grid.nodes(d), axis.nodes), (name, d)
        assert _same_bits(grid.interior_faces(d), axis.faces), (name, d)


def test_the_meshes_are_what_their_names_say() -> None:
    """The product grids have SOLID cells; the stretched one is not uniform."""
    for name in ("80x30", "160x60", "320x120", "160x60-stretched"):
        assert (_mesh(name).cell_type == SOLID).any(), name
    assert not (_mesh("40x15-uniform-open").cell_type == SOLID).any()
    widths = np.diff(_mesh("160x60-stretched").x)
    assert widths.max() / widths.min() > 1.5
    for name in VENT_MESHES:
        mesh = _mesh(name)
        vents = vent_face_masks(
            mesh, vent_runs(mesh, _config(name), BoundaryRegistry(_config(name)))
        )
        assert vents.normal_u.any() and vents.normal_v.any(), name


@dataclasses.dataclass(frozen=True, eq=False)
class Call:
    """The arguments of one core function, by name and in the order it takes them."""

    def replaced(self, **changes: np.ndarray | int | None) -> Self:
        """These arguments with some replaced, by name."""
        return dataclasses.replace(self, **changes)

    def invoke(self, function: Callable[..., object]) -> object:
        """Call ``function`` with these arguments, in field order."""
        return function(*(getattr(self, f.name) for f in dataclasses.fields(self)))


@dataclasses.dataclass(frozen=True, eq=False)
class GridArguments(Call):
    """The arguments of ``Grid2``."""

    x: np.ndarray
    xc: np.ndarray
    y: np.ndarray
    yc: np.ndarray
    cell_type: np.ndarray
    solid: int


def _grid_arrays() -> GridArguments:
    """A small consistent set of Grid2's arguments, [ny, nx] = [3, 4]."""
    return GridArguments(
        x=np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
        xc=np.array([0.5, 1.5, 2.5, 3.5]),
        y=np.array([0.0, 1.0, 2.0, 3.0]),
        yc=np.array([0.5, 1.5, 2.5]),
        cell_type=np.zeros((3, 4), dtype=np.int32),
        solid=SOLID,
    )


# What the refusal tests below build their wrong arrays from. Each wrong array
# is wrong in exactly one way (a shape, a layout, a dtype, an alignment or a
# dimension count) and the test names the refusal it expects by its message, so a
# case that another check refuses first fails instead of passing for the wrong
# reason. nanobind refuses a wrong dtype, layout or dimension with this text.
NANOBIND_REFUSAL = "incompatible function arguments"
UNALIGNED_REFUSAL = "not aligned"
SHARED_REFUSAL = "shares memory"


class Refusal(NamedTuple):
    """One wrong array: the argument it replaces, how it is changed, the refusal's text."""

    key: str
    change: Callable[[np.ndarray], np.ndarray]
    message: str
    name: str


def refusals(cases: list[Refusal]) -> pytest.MarkDecorator:
    """Parametrize a test over wrong-array cases, with each case's name as its id."""
    return pytest.mark.parametrize(
        ("key", "change", "message"),
        [case[:3] for case in cases],
        ids=[case.name for case in cases],
    )


def _strided(a: np.ndarray) -> np.ndarray:
    """The same values and shape in a view that is not C-contiguous (a step of two)."""
    return np.repeat(a, 2, axis=-1)[..., ::2]


def _fortran(a: np.ndarray) -> np.ndarray:
    """The same values and shape in Fortran order, as a transposed copy's view is."""
    return np.asfortranarray(a)


def _fewer_rows(a: np.ndarray) -> np.ndarray:
    """A C-contiguous array with one row fewer: the right layout and dtype, a wrong shape."""
    return a[:-1]


def _fewer_columns(a: np.ndarray) -> np.ndarray:
    """A C-contiguous array with one column fewer, in a buffer twice as long as ``a``.

    The spare room means that an ``out`` whose shape check is missing is written
    past its own shape but not past its allocation, so the test fails with
    "DID NOT RAISE" and the run goes on, where a tight array would corrupt the heap.
    """
    room = np.zeros(2 * a.size, dtype=a.dtype)
    view = room[: a.shape[0] * (a.shape[1] - 1)].reshape(a.shape[0], a.shape[1] - 1)
    view[...] = a[:, :-1]
    return view


def _unaligned_like(a: np.ndarray) -> np.ndarray:
    """A copy of an array whose data pointer is one byte past a multiple of eight."""
    raw = np.zeros(a.nbytes + 16, dtype=np.uint8)
    start = (1 - raw.ctypes.data) % 8
    out = raw[start : start + a.nbytes].view(a.dtype).reshape(a.shape)
    out[...] = a
    return out


def _overlapping(
    out_shape: tuple[int, ...], other: np.ndarray, end: str
) -> tuple[np.ndarray, np.ndarray]:
    """A float64 ``out`` of ``out_shape`` and a copy of ``other`` that overlap in one buffer.

    With ``end`` "tail" the copy starts at the last element of ``out``, so it
    overlaps ``out``'s tail. With "head" ``out`` starts where the copy's last
    element (for a bool array, its last byte) is, rounded down to eight bytes,
    so ``out`` overlaps the copy's tail. Either way each array reaches into the
    other and neither contains it.
    """
    out_bytes = int(np.prod(out_shape)) * 8
    if end == "tail":
        out_start, other_start = 0, out_bytes - other.itemsize
    else:
        out_start, other_start = (other.nbytes - other.itemsize) // 8 * 8, 0
    size = max(out_start + out_bytes, other_start + other.nbytes)
    buffer = np.zeros(size + 8, dtype=np.uint8)
    # Start on a multiple of eight, so the offsets above keep a float64 view aligned.
    raw = buffer[(-buffer.ctypes.data) % 8 :]
    out = raw[out_start : out_start + out_bytes].view(np.float64).reshape(out_shape)
    shared = raw[other_start : other_start + other.nbytes].view(other.dtype)
    shared = shared.reshape(other.shape)
    shared[...] = other
    return out, shared


def test_grid_accepts_consistent_arrays() -> None:
    """The arrays _grid_arrays builds make a grid, so each refusal below is its change's."""
    grid = _grid_arrays().invoke(_cfdcore.Grid2)
    assert grid.shape == (3, 4)


# The Grid2 binding checks cell_type against the centres itself, and the grid's
# own check refuses faces that are not one more than their centres (a TypeError
# either way). The cell types' transposed copy has the cells' count, so only the
# binding's shape check refuses it.
GRID_REFUSALS = [
    Refusal("x", lambda a: np.arange(6.0), "faces are not one more", "faces-too-long"),
    Refusal(
        "cell_type",
        lambda a: a.T.copy(),
        r"cell_type is not \[len\(yc\), len\(xc\)\]",
        "cell-type-transposed",
    ),
    *(
        Refusal(
            key, _strided, NANOBIND_REFUSAL, f"{key.replace('_', '-')}-non-contiguous"
        )
        for key in ("x", "xc", "y", "yc", "cell_type")
    ),
    Refusal("cell_type", _fortran, NANOBIND_REFUSAL, "cell-type-fortran-order"),
    Refusal("cell_type", lambda a: a.reshape(-1), NANOBIND_REFUSAL, "cell-type-flat"),
    *(
        Refusal(key, lambda a: a.astype(np.float32), NANOBIND_REFUSAL, f"{key}-float32")
        for key in ("x", "xc", "y", "yc")
    ),
    Refusal(
        "cell_type", lambda a: a.astype(np.int64), NANOBIND_REFUSAL, "cell-type-int64"
    ),
    Refusal(
        "cell_type",
        lambda a: a.astype(np.float64),
        NANOBIND_REFUSAL,
        "cell-type-float64",
    ),
    *(
        Refusal(
            key,
            _unaligned_like,
            UNALIGNED_REFUSAL,
            f"{key.replace('_', '-')}-unaligned",
        )
        for key in ("x", "xc", "y", "yc", "cell_type")
    ),
]


@refusals(GRID_REFUSALS)
def test_grid_refuses_wrong_arrays(
    key: str, change: Callable[[np.ndarray], np.ndarray], message: str
) -> None:
    """A wrong shape, dtype, layout, dimension or alignment is a TypeError at the binding."""
    arguments = _grid_arrays()
    wrong = arguments.replaced(**{key: change(getattr(arguments, key))})
    with pytest.raises(TypeError, match=message):
        wrong.invoke(_cfdcore.Grid2)


def test_grid_refuses_a_direction_without_a_cell() -> None:
    """A direction with no cell is a TypeError from the grid's own check.

    The cell types have no cell either, so the binding's shape check passes
    and it is ``Grid::check`` that refuses.
    """
    arguments = _grid_arrays().replaced(
        x=np.array([0.0]), xc=np.zeros(0), cell_type=np.zeros((3, 0), dtype=np.int32)
    )
    with pytest.raises(TypeError, match="has no cell"):
        arguments.invoke(_cfdcore.Grid2)


# Each change breaks one thing along a direction: (the array it changes, the
# index, the new value from a copy of the faces and of the centres, which the
# default's are 0..n and the midpoints). A face that repeats or decreases; a NaN
# face; an infinite end face (the order holds, so only the finiteness test
# refuses it); a centre that repeats, decreases or is NaN; a centre at or beyond
# an end face (the centres are in order, so only the end clause refuses it).
BAD_COORDINATES = {
    "face-repeats": ("faces", 1, lambda f, c: f[0]),
    "face-decreases": ("faces", 1, lambda f, c: f[0] - 1.0),
    "face-nan": ("faces", 1, lambda f, c: np.nan),
    "last-face-infinite": ("faces", -1, lambda f, c: np.inf),
    "first-face-infinite": ("faces", 0, lambda f, c: -np.inf),
    "centre-repeats": ("centres", 1, lambda f, c: c[0]),
    "centre-decreases": ("centres", 1, lambda f, c: c[0] - 0.1),
    "centre-nan": ("centres", 0, lambda f, c: np.nan),
    "centre-at-low-face": ("centres", 0, lambda f, c: f[0]),
    "centre-at-high-face": ("centres", -1, lambda f, c: f[-1]),
    "centre-below-low-face": ("centres", 0, lambda f, c: f[0] - 1.0),
    "centre-past-high-face": ("centres", -1, lambda f, c: f[-1] + 1.0),
}


@pytest.mark.parametrize("direction", [0, 1])
@pytest.mark.parametrize("kind", list(BAD_COORDINATES))
def test_grid_refuses_coordinates_that_are_not_finite_and_increasing(
    direction: int, kind: str
) -> None:
    """Faces or nodes out of order or not finite are a ValueError naming the direction."""
    arguments = _grid_arrays()
    faces_key, centres_key = (("x", "xc"), ("y", "yc"))[direction]
    faces = getattr(arguments, faces_key).copy()
    centres = getattr(arguments, centres_key).copy()
    target, index, value = BAD_COORDINATES[kind]
    (faces if target == "faces" else centres)[index] = value(faces, centres)
    wrong = arguments.replaced(**{faces_key: faces, centres_key: centres})
    with pytest.raises(ValueError, match=f"along {'xy'[direction]} "):
        wrong.invoke(_cfdcore.Grid2)


def test_grid_names_the_lower_direction_when_both_are_bad() -> None:
    """With bad coordinates along x and y the refusal names x, the first checked."""
    arguments = _grid_arrays()
    x, y = arguments.x.copy(), arguments.y.copy()
    x[1], y[1] = x[0], y[0]
    with pytest.raises(ValueError, match="along x "):
        arguments.replaced(x=x, y=y).invoke(_cfdcore.Grid2)


def test_grid_refuses_a_direction_it_does_not_have() -> None:
    """A direction other than 0 or 1 is a ValueError."""
    grid = _grid_arrays().invoke(_cfdcore.Grid2)
    with pytest.raises(ValueError):
        grid.nodes(2)


# ---- The scalar scheme's explicit flux (prompt 87) ---------------------------

# The planted fields: random values of both signs over twelve decades; a
# uniform field (exact under every rule); a field of a few levels, so many
# downstream differences are zero; a smooth field with steps and spikes beside
# and inside the SOLID cells; small values with signed zeros, where the
# sign of a zero psi reaches the face value; and -0.0 among negative values and
# a positive one, where the far-solid rule's quick = c_c line decides that sign.
FIELDS = [
    "random",
    "uniform",
    "levels",
    "steps-spikes",
    "signed-zeros",
    "negatives-and-zeros",
]


def _planted_field(kind: str, mesh: Mesh, rng: np.random.Generator) -> np.ndarray:
    """A planted cell field of one of FIELDS on a mesh, [ny, nx]."""
    shape = mesh.cell_type.shape
    if kind == "random":
        return _values(rng, shape)
    if kind == "uniform":
        return np.full(shape, 3.7e4)
    if kind == "levels":
        return rng.choice(np.array([-2.5, 0.0, 1.0, 1.0e3]), size=shape)
    if kind == "steps-spikes":
        x, y = np.meshgrid(mesh.xc, mesh.yc)
        field = 1.0e4 * np.exp(-((x - 4.0) ** 2) - (y - 1.5) ** 2) + np.where(
            x > 3.0, 50.0, 0.0
        )
        solid = mesh.cell_type == SOLID
        beside = np.zeros_like(solid)
        beside[:, 1:] |= solid[:, :-1]
        beside[:, :-1] |= solid[:, 1:]
        beside[1:, :] |= solid[:-1, :]
        beside[:-1, :] |= solid[1:, :]
        field = np.where(solid, 1.0e9, field)
        field = np.where(beside & ~solid, -3.0e5, field)
        spikes = rng.random(shape) < 0.02
        return np.where(spikes, 7.0e7, field)
    if kind == "signed-zeros":
        return rng.choice(np.array([0.0, -0.0, 1.0e-300, -2.0e-310, 5.0]), size=shape)
    if kind == "negatives-and-zeros":
        return rng.choice(np.array([-0.0, -1.0, -3.0, 2.0]), size=shape)
    raise ValueError(kind)


def _face_shape(mesh: Mesh, d: int) -> tuple[int, int]:
    """The shape of the face field normal to direction d."""
    ny, nx = mesh.cell_type.shape
    return (ny, nx + 1) if d == 0 else (ny + 1, nx)


def _planted_flux(shape: tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    """A volume flux of both signs over twelve decades, zero (and -0.0) on some faces."""
    flux = _values(rng, shape)
    pick = rng.random(shape)
    flux[pick < 0.1] = 0.0
    flux[(pick >= 0.1) & (pick < 0.13)] = -0.0
    return flux


def _vent_masks(name: str) -> tuple[np.ndarray, np.ndarray]:
    """The vents file's normal vent faces on a VENT_MESHES grid, u and v."""
    mesh = _mesh(name)
    config = _config(name)
    vents = vent_face_masks(mesh, vent_runs(mesh, config, BoundaryRegistry(config)))
    return vents.normal_u, vents.normal_v


@functools.cache
def _scheme(name: str) -> "_cfdcore.ScalarScheme2":
    """The core's scalar scheme on one of the parity meshes, built once."""
    return _cfdcore.ScalarScheme2(_grid(_mesh(name)))


def _reference_flux(
    mesh: Mesh,
    d: int,
    c: np.ndarray,
    flux: np.ndarray,
    inflow: np.ndarray,
    upwind: bool,
    vents: np.ndarray | None,
) -> np.ndarray:
    """``advective_flux`` as the transport solver calls it, y on the transposed arrays."""
    axis_x, axis_y = mesh_axes(mesh)
    # The planted differences reach 1e-310, where r overflows to an infinity;
    # the clamp takes it to 0 or 2 in both, and NumPy's warning is the only noise.
    with np.errstate(over="ignore"):
        if d == 0:
            return advective_flux(c, flux, inflow, axis_x, upwind, vents)
        v = None if vents is None else np.ascontiguousarray(vents.T)
        return advective_flux(c.T, flux.T, inflow.T, axis_y, upwind, v).T


def _core_flux(
    name: str,
    d: int,
    c: np.ndarray,
    flux: np.ndarray,
    inflow: np.ndarray,
    upwind: bool,
    vents: np.ndarray | None,
) -> np.ndarray:
    """The core's advective_flux on the fields in their own layout."""
    out = np.full(flux.shape, np.nan)
    _cfdcore.advective_flux(_scheme(name), d, c, flux, inflow, upwind, out, vents)
    return out


FLUX_CASES = [
    pytest.param(name, vents, id=f"{name}-{vents}")
    for name, vents in [
        *((name, "none") for name in MESHES),
        *((name, "random-mask") for name in ("80x30", "160x60-stretched")),
        *((name, "none") for name in VENT_MESHES),
        *((name, "vents") for name in VENT_MESHES),
    ]
]


@pytest.mark.parametrize(("name", "vents"), FLUX_CASES)
@pytest.mark.parametrize("d", [0, 1])
@pytest.mark.parametrize("upwind", [False, True])
@pytest.mark.parametrize("kind", FIELDS)
def test_advective_flux_equals_the_reference_bit_for_bit(
    name: str, vents: str, d: int, upwind: bool, kind: str
) -> None:
    """advective_flux is the reference's bit for bit (prediction (a), planted inputs)."""
    rng = np.random.default_rng([SEED, d, int(upwind), FIELDS.index(kind), len(name)])
    mesh = _mesh(name)
    shape = _face_shape(mesh, d)
    c = _planted_field(kind, mesh, rng)
    flux = _planted_flux(shape, rng)
    inflow = _values(rng, shape)
    mask = None
    if vents == "vents":
        mask = _vent_masks(name)[d]
    elif vents == "random-mask":
        # Beyond the contract (a vent sits beside a SOLID cell), but the
        # arithmetic is the same on any face, so any mask must match.
        mask = rng.random(shape) < 0.2
    reference = _reference_flux(mesh, d, c, flux, inflow, upwind, mask)
    core = _core_flux(name, d, c, flux, inflow, upwind, mask)
    differ = np.flatnonzero(reference.view(np.uint64) != core.view(np.uint64))
    assert differ.size == 0, (
        differ[:10],
        reference.flat[differ[:3]],
        core.flat[differ[:3]],
    )


def test_the_planted_inputs_reach_every_rule() -> None:
    """The planted cases exercise the far-solid rule, both vent cases and the zero branches.

    Without these the bitwise comparison above could pass on rules no face
    reaches. Counted on the 80x30 vents mesh with the random field.
    """
    name = "80x30-vents"
    mesh = _mesh(name)
    solid = mesh.cell_type == SOLID
    rng = np.random.default_rng(SEED)
    counts = {"far_solid": 0, "vent_entering": 0, "vent_leaving": 0}
    for d, vents in enumerate(_vent_masks(name)):
        flux = _planted_flux(_face_shape(mesh, d), rng)
        interior = flux[:, 1:-1] if d == 0 else flux[1:-1, :].T
        s = (
            np.pad(solid, ((0, 0), (1, 1)))
            if d == 0
            else np.pad(solid, ((1, 1), (0, 0))).T
        )
        k = np.arange(1, interior.shape[1] + 1)
        positive = interior > 0.0
        far = np.where(positive, s[:, k - 1], s[:, k + 2])
        counts["far_solid"] += int((far & (interior != 0.0)).sum())
        v = vents[:, 1:-1] if d == 0 else vents[1:-1, :].T
        entering = (s[:, k] & (interior > 0.0)) | (s[:, k + 1] & (interior < 0.0))
        counts["vent_entering"] += int((v & entering).sum())
        counts["vent_leaving"] += int((v & ~entering & (interior != 0.0)).sum())
    assert all(n > 0 for n in counts.values()), counts
    # The levels field has zero downstream differences; the signed-zeros
    # field makes a -0.0 flux times a face value of either sign.
    c = _planted_field("levels", mesh, rng)
    assert (np.diff(c, axis=1) == 0.0).mean() > 0.2
    c = _planted_field("signed-zeros", mesh, rng)
    flux = _planted_flux(_face_shape(mesh, 0), rng)
    out = _reference_flux(mesh, 0, c, flux, _values(rng, flux.shape), False, None)
    assert np.signbit(out[out == 0.0]).any() and (~np.signbit(out[out == 0.0])).any()
    # The far-solid rule's `quick = c_c` line changes a bit on the last field,
    # which the signed-zeros field does not reach.
    c = _planted_field("negatives-and-zeros", mesh, rng)
    flux = _planted_flux(_face_shape(mesh, 0), rng)
    inflow = _values(rng, flux.shape)
    assert _faces_the_far_solid_quick_line_decides(mesh, c, flux, inflow) > 0


def _faces_the_far_solid_quick_line_decides(
    mesh: Mesh, c: np.ndarray, flux: np.ndarray, inflow: np.ndarray
) -> int:
    """The x faces whose value changes a bit if the far-solid rule leaves ``quick`` alone.

    ``advective_flux`` sets both ``c_up`` and ``quick`` to the upstream value
    where the far node is SOLID. This forms the clamp's inputs as it does
    (QUICK, no vents) and counts the faces where setting ``c_up`` alone gives
    another bit pattern. ``c_up = c_c`` makes r zero, so ``quick`` reaches
    the face value only through the sign of a zero psi.
    """
    axis = mesh_axes(mesh)[0]
    c_low = np.where(flux[:, 0] > 0.0, inflow[:, 0], c[:, 0])
    c_high = np.where(flux[:, -1] < 0.0, inflow[:, -1], c[:, -1])
    ext = np.concatenate([c_low[:, None], c, c_high[:, None]], axis=1)
    k = np.asarray(axis.left)
    positive = flux[:, 1:-1] > 0.0
    c_c = np.where(positive, ext[:, k], ext[:, k + 1])
    c_d = np.where(positive, ext[:, k + 1], ext[:, k])
    c_up = np.where(positive, ext[:, k - 1], ext[:, k + 2])
    far_solid = np.where(positive, axis.solid_ext[:, k - 1], axis.solid_ext[:, k + 2])
    quick = quick_face_values(ext, axis.nodes, axis.left, axis.faces, positive)
    c_up = np.where(far_solid, c_c, c_up)
    with np.errstate(over="ignore"):
        with_line = limited_face_values(c_up, c_c, c_d, np.where(far_solid, c_c, quick))
        without = limited_face_values(c_up, c_c, c_d, quick)
    differ = with_line.view(np.uint64) != without.view(np.uint64)
    return int((far_solid & (flux[:, 1:-1] != 0.0) & differ).sum())


@pytest.mark.parametrize("name", list(MESHES))
@pytest.mark.parametrize("d", [0, 1])
def test_quick_weights_at_setup_equal_the_per_call_weights(name: str, d: int) -> None:
    """The weights stored once equal quick_face_values' per-call weights (prediction (c))."""
    axis = mesh_axes(_mesh(name))[d]
    nodes = axis.nodes
    n = nodes.shape[0]
    # quick_face_values' own indexing (src/momentum.py).
    k = np.asarray(axis.left)
    far_pos = np.where(k - 1 >= 0, k - 1, k + 2)
    far_neg = np.where(k + 2 <= n - 1, k + 2, k - 1)
    w_pos = _lagrange_weights(nodes[far_pos], nodes[k], nodes[k + 1], axis.faces)
    w_neg = _lagrange_weights(nodes[far_neg], nodes[k + 1], nodes[k], axis.faces)
    core = _scheme(name).weights(d)
    for i, reference in enumerate((*w_pos, *w_neg)):
        assert _same_bits(core[i], reference), (name, d, i)


@pytest.mark.parametrize("name", list(MESHES))
@pytest.mark.parametrize("d", [0, 1])
@pytest.mark.parametrize("kind", ["random", "levels", "signed-zeros"])
def test_quick_face_values_equal_the_reference_bit_for_bit(
    name: str, d: int, kind: str
) -> None:
    """quick_face_values on the field and its boundary nodes is the reference's bit for bit."""
    rng = np.random.default_rng([SEED, d, FIELDS.index(kind), len(name), 7])
    mesh = _mesh(name)
    axis = mesh_axes(mesh)[d]
    c = _planted_field(kind, mesh, rng)
    lines = c.shape[0] if d == 0 else c.shape[1]
    c_low = _values(rng, lines)
    c_high = _values(rng, lines)
    positive = rng.random(_face_shape(mesh, d)) < 0.5
    out = np.full(positive.shape, np.nan)
    _cfdcore.quick_face_values(_scheme(name), d, c, c_low, c_high, positive, out)
    c_t, pos_t, core = (c, positive[:, 1:-1], out[:, 1:-1])
    if d == 1:
        c_t, pos_t, core = (c.T, positive[1:-1, :].T, out[1:-1, :].T)
    ext = np.concatenate([c_low[:, None], c_t, c_high[:, None]], axis=1)
    reference = quick_face_values(ext, axis.nodes, axis.left, axis.faces, pos_t)
    assert _same_bits(core, reference)
    # The boundary faces are left as they were.
    ends = out[:, [0, -1]] if d == 0 else out[[0, -1], :]
    assert np.isnan(ends).all()


@pytest.mark.parametrize(
    "shape",
    [(1,), (7,), (1000,), (30, 79), (119, 320)],
    ids=lambda shape: "x".join(map(str, shape)),
)
def test_limited_face_values_equal_the_reference_bit_for_bit(
    shape: tuple[int, ...],
) -> None:
    """The UMIST clamp is the reference's bit for bit, ties and signed zeros included."""
    rng = np.random.default_rng([SEED, *shape])
    levels = np.array([-1.0, -0.0, 0.0, 2.0, 1.0e-300])

    def planted() -> np.ndarray:
        # Half random over twelve decades, half a few levels, so that zero
        # differences, ties in the min and max, and signed zeros all occur.
        return np.where(
            rng.random(shape) < 0.5, _values(rng, shape), rng.choice(levels, shape)
        )

    c_up, c_c, c_d, quick = planted(), planted(), planted(), planted()
    out = np.full(shape, np.nan)
    _cfdcore.limited_face_values(c_up, c_c, c_d, quick, out)
    with np.errstate(over="ignore"):
        reference = limited_face_values(c_up, c_c, c_d, quick)
    assert _same_bits(out, reference)


def test_numpy_min_and_max_return_the_second_of_a_tie() -> None:
    """The core copies NumPy's tie rule for np.minimum and np.maximum; this pins it.

    The clamp's psi is ``np.maximum(0.0, ...)`` of a fold of np.minimum, and a
    zero psi's sign reaches the face value when the upstream value is -0.0.
    If a NumPy build returned the first of a tie this test, not the bitwise
    tests' last bits, says why.
    """
    for n in (1, 3, 8, 17, 1000):
        zero, minus = np.zeros(n), np.full(n, -0.0)
        assert np.signbit(np.minimum(zero, minus)).all()
        assert not np.signbit(np.minimum(minus, zero)).any()
        assert np.signbit(np.maximum(0.0, minus)).all()
        assert not np.signbit(np.maximum(minus, zero)).any()


@dataclasses.dataclass(frozen=True, eq=False)
class FluxArguments(Call):
    """The arguments of ``advective_flux`` and ``advective_flux_overhead``."""

    scheme: "_cfdcore.ScalarScheme2"
    direction: int
    c: np.ndarray
    flux: np.ndarray
    inflow: np.ndarray
    upwind: bool
    out: np.ndarray
    vents: np.ndarray | None


def _flux_arguments(d: int = 0) -> FluxArguments:
    """Consistent advective_flux arguments on the 40x15 open mesh."""
    mesh = _mesh("40x15-uniform-open")
    shape = _face_shape(mesh, d)
    return FluxArguments(
        scheme=_scheme("40x15-uniform-open"),
        direction=d,
        c=np.ones(mesh.cell_type.shape),
        flux=np.ones(shape),
        inflow=np.ones(shape),
        upwind=False,
        out=np.zeros(shape),
        vents=np.zeros(shape, dtype=bool),
    )


def test_advective_flux_accepts_consistent_arguments() -> None:
    """The arguments _flux_arguments builds are accepted, so each refusal is its change's."""
    for d in (0, 1):
        arguments = _flux_arguments(d)
        arguments.invoke(_cfdcore.advective_flux)
        assert (arguments.out == 1.0).all()
        arguments.invoke(_cfdcore.advective_flux_overhead)


def _wrong_dtype(a: np.ndarray) -> np.ndarray:
    """The same values and shape in the dtype the binding does not take."""
    return a.astype(np.uint8 if a.dtype == np.bool_ else np.float32)


def _refusal_cases(
    keys: tuple[str, ...], shaped: tuple[str, ...], aligned: tuple[str, ...]
) -> list[Refusal]:
    """Wrong-array cases for the arguments ``keys``, each wrong in exactly one way.

    ``shaped`` are the arguments the binding checks against a shape of its own
    (a message ending ``<name> is not [rows, columns]``); ``aligned`` those it
    checks the alignment of, which a bool array has no use for.
    """
    cases = []
    for key in keys:
        name = key.replace("_", "-")
        if key in shaped:
            shape_refusal = rf": {key} is not \["
            cases += [
                Refusal(key, _fewer_rows, shape_refusal, f"{name}-fewer-rows"),
                Refusal(key, _fewer_columns, shape_refusal, f"{name}-fewer-columns"),
            ]
        cases += [
            Refusal(key, _strided, NANOBIND_REFUSAL, f"{name}-non-contiguous"),
            Refusal(key, _wrong_dtype, NANOBIND_REFUSAL, f"{name}-wrong-dtype"),
        ]
        if key in aligned:
            cases.append(
                Refusal(key, _unaligned_like, UNALIGNED_REFUSAL, f"{name}-unaligned")
            )
    return cases


FLUX_REFUSALS = [
    *_refusal_cases(
        ("c", "flux", "inflow", "out", "vents"),
        shaped=("c", "flux", "inflow", "out", "vents"),
        aligned=("c", "flux", "inflow", "out"),
    ),
    # The matrices also come in Fortran order, as a transposed field does.
    *(
        Refusal(key, _fortran, NANOBIND_REFUSAL, f"{key}-fortran-order")
        for key in ("c", "flux", "inflow", "out", "vents")
    ),
    Refusal("c", lambda a: a.reshape(-1), NANOBIND_REFUSAL, "c-flat"),
]


@pytest.mark.parametrize("d", [0, 1])
@refusals(FLUX_REFUSALS)
@pytest.mark.parametrize("function", ["advective_flux", "advective_flux_overhead"])
def test_advective_flux_refuses_wrong_arrays(
    function: str,
    key: str,
    change: Callable[[np.ndarray], np.ndarray],
    message: str,
    d: int,
) -> None:
    """A wrong shape, dtype, layout, dimension or alignment is a TypeError at the binding.

    The overhead function makes the same checks, so it refuses the same arrays.
    """
    arguments = _flux_arguments(d)
    wrong = arguments.replaced(**{key: change(getattr(arguments, key))})
    with pytest.raises(TypeError, match=message):
        wrong.invoke(getattr(_cfdcore, function))


def test_advective_flux_refuses_a_read_only_out() -> None:
    """A read-only out is a TypeError: the core could not write the flux to it."""
    arguments = _flux_arguments()
    arguments.out.flags.writeable = False
    with pytest.raises(TypeError, match=NANOBIND_REFUSAL):
        arguments.invoke(_cfdcore.advective_flux)


@pytest.mark.parametrize("end", ["tail", "head"])
@pytest.mark.parametrize("operand", ["c", "flux", "inflow", "vents"])
def test_advective_flux_refuses_an_out_that_overlaps_an_input(
    operand: str, end: str
) -> None:
    """An out sharing one element with a single input is a TypeError, whichever end.

    One input at a time, so each operand's check is pinned alone.
    """
    arguments = _flux_arguments()
    out, shared = _overlapping(arguments.out.shape, getattr(arguments, operand), end)
    wrong = arguments.replaced(out=out, **{operand: shared})
    with pytest.raises(TypeError, match=SHARED_REFUSAL):
        wrong.invoke(_cfdcore.advective_flux)


@pytest.mark.parametrize("operand", ["flux", "inflow"])
def test_advective_flux_refuses_an_out_that_is_an_input(operand: str) -> None:
    """An out that is the flux or the inflow itself is a TypeError."""
    arguments = _flux_arguments()
    wrong = arguments.replaced(out=getattr(arguments, operand))
    with pytest.raises(TypeError, match=SHARED_REFUSAL):
        wrong.invoke(_cfdcore.advective_flux)


@pytest.mark.parametrize("out_first", [True, False])
def test_advective_flux_accepts_an_out_that_touches_no_input(out_first: bool) -> None:
    """Arrays cut from one buffer, side by side and overlapping none, are accepted.

    The out comes below every input, then above every input. An aliasing
    refusal that fires on neighbours, or that tests only one end of the
    overlap, would refuse one of the two orders and pass the tests above.
    """
    arguments = _flux_arguments()
    names = (
        ("out", "c", "flux", "inflow") if out_first else ("c", "flux", "inflow", "out")
    )
    buffer = np.zeros(sum(getattr(arguments, name).size for name in names))
    pieces: dict[str, np.ndarray] = {}
    start = 0
    for name in names:
        shape = getattr(arguments, name).shape
        pieces[name] = buffer[start : start + int(np.prod(shape))].reshape(shape)
        start += int(np.prod(shape))
    for name in ("c", "flux", "inflow"):
        pieces[name][...] = getattr(arguments, name)
    arguments.replaced(vents=None, **pieces).invoke(_cfdcore.advective_flux)
    assert (pieces["out"] == 1.0).all()


def test_advective_flux_refuses_a_direction_it_does_not_have() -> None:
    """A direction other than 0 or 1 is a ValueError."""
    with pytest.raises(ValueError):
        _flux_arguments().replaced(direction=2).invoke(_cfdcore.advective_flux)


@dataclasses.dataclass(frozen=True, eq=False)
class QuickArguments(Call):
    """The arguments of ``quick_face_values``."""

    scheme: "_cfdcore.ScalarScheme2"
    direction: int
    c: np.ndarray
    c_low: np.ndarray
    c_high: np.ndarray
    positive: np.ndarray
    out: np.ndarray


def _quick_arguments(d: int = 0) -> QuickArguments:
    """Consistent quick_face_values arguments on the 40x15 open mesh."""
    mesh = _mesh("40x15-uniform-open")
    shape = _face_shape(mesh, d)
    # One line per row along x and per column along y: shape[d] of [ny, nx].
    lines = mesh.cell_type.shape[d]
    return QuickArguments(
        scheme=_scheme("40x15-uniform-open"),
        direction=d,
        c=np.ones(mesh.cell_type.shape),
        c_low=np.ones(lines),
        c_high=np.ones(lines),
        positive=np.ones(shape, dtype=bool),
        out=np.zeros(shape),
    )


def test_quick_face_values_accepts_consistent_arguments() -> None:
    """The arguments _quick_arguments builds are accepted, so each refusal is its change's."""
    for d in (0, 1):
        _quick_arguments(d).invoke(_cfdcore.quick_face_values)


QUICK_REFUSALS = [
    *_refusal_cases(
        ("c", "positive", "out"),
        shaped=("c", "positive", "out"),
        aligned=("c", "out"),
    ),
    # The boundary values are vectors: one per line, which the binding counts.
    Refusal("c_low", lambda a: a[:-1], "hold one value per line", "c-low-too-short"),
    Refusal("c_high", lambda a: a[:-1], "hold one value per line", "c-high-too-short"),
    *(
        Refusal(key, change, message, f"{key.replace('_', '-')}-{label}")
        for key in ("c_low", "c_high")
        for label, change, message in (
            ("non-contiguous", _strided, NANOBIND_REFUSAL),
            ("wrong-dtype", _wrong_dtype, NANOBIND_REFUSAL),
            ("unaligned", _unaligned_like, UNALIGNED_REFUSAL),
        )
    ),
    *(
        Refusal(key, _fortran, NANOBIND_REFUSAL, f"{key}-fortran-order")
        for key in ("c", "positive", "out")
    ),
    Refusal("c", lambda a: a.reshape(-1), NANOBIND_REFUSAL, "c-flat"),
]


@pytest.mark.parametrize("d", [0, 1])
@refusals(QUICK_REFUSALS)
def test_quick_face_values_refuses_wrong_arrays(
    key: str, change: Callable[[np.ndarray], np.ndarray], message: str, d: int
) -> None:
    """A wrong shape, dtype, layout, dimension or alignment is a TypeError at the binding."""
    arguments = _quick_arguments(d)
    wrong = arguments.replaced(**{key: change(getattr(arguments, key))})
    with pytest.raises(TypeError, match=message):
        wrong.invoke(_cfdcore.quick_face_values)


@pytest.mark.parametrize("end", ["tail", "head"])
@pytest.mark.parametrize("operand", ["c", "c_low", "c_high", "positive"])
def test_quick_face_values_refuses_an_out_that_overlaps_an_input(
    operand: str, end: str
) -> None:
    """An out sharing one element with a single input is a TypeError, whichever end."""
    arguments = _quick_arguments()
    out, shared = _overlapping(arguments.out.shape, getattr(arguments, operand), end)
    wrong = arguments.replaced(out=out, **{operand: shared})
    with pytest.raises(TypeError, match=SHARED_REFUSAL):
        wrong.invoke(_cfdcore.quick_face_values)


def test_quick_face_values_refuses_a_direction_it_does_not_have() -> None:
    """A direction other than 0 or 1 is a ValueError."""
    with pytest.raises(ValueError):
        _quick_arguments().replaced(direction=2).invoke(_cfdcore.quick_face_values)


@dataclasses.dataclass(frozen=True, eq=False)
class LimitedArguments(Call):
    """The arguments of ``limited_face_values``."""

    c_up: np.ndarray
    c_c: np.ndarray
    c_d: np.ndarray
    quick: np.ndarray
    out: np.ndarray


def _limited_arguments() -> LimitedArguments:
    """Consistent limited_face_values arguments: five arrays of one [4, 5] shape."""
    return LimitedArguments(
        c_up=np.ones((4, 5)),
        c_c=np.ones((4, 5)),
        c_d=np.ones((4, 5)),
        quick=np.ones((4, 5)),
        out=np.zeros((4, 5)),
    )


def test_limited_face_values_accepts_consistent_arguments() -> None:
    """The arguments _limited_arguments builds are accepted, so each refusal is its change's."""
    _limited_arguments().invoke(_cfdcore.limited_face_values)


SHAPES_DIFFER = "shapes differ"
LIMITED_REFUSALS = [
    Refusal(key, change, message, f"{key.replace('_', '-')}-{label}")
    for key in ("c_up", "c_c", "c_d", "quick", "out")
    for label, change, message in (
        ("fewer-rows", _fewer_rows, SHAPES_DIFFER),
        ("flat", lambda a: a.reshape(-1), SHAPES_DIFFER),
        ("non-contiguous", _strided, NANOBIND_REFUSAL),
        ("fortran-order", _fortran, NANOBIND_REFUSAL),
        ("wrong-dtype", _wrong_dtype, NANOBIND_REFUSAL),
        ("unaligned", _unaligned_like, UNALIGNED_REFUSAL),
    )
]


@refusals(LIMITED_REFUSALS)
def test_limited_face_values_refuses_wrong_arrays(
    key: str, change: Callable[[np.ndarray], np.ndarray], message: str
) -> None:
    """A wrong shape, dimension, dtype, layout or alignment is a TypeError at the binding."""
    arguments = _limited_arguments()
    wrong = arguments.replaced(**{key: change(getattr(arguments, key))})
    with pytest.raises(TypeError, match=message):
        wrong.invoke(_cfdcore.limited_face_values)


@pytest.mark.parametrize("end", ["tail", "head", "same"])
@pytest.mark.parametrize("operand", ["c_up", "c_c", "c_d", "quick"])
def test_limited_face_values_refuses_an_out_that_overlaps_an_input(
    operand: str, end: str
) -> None:
    """An out sharing memory with a single input is a TypeError: one element or all of it."""
    arguments = _limited_arguments()
    if end == "same":
        wrong = arguments.replaced(out=getattr(arguments, operand))
    else:
        out, shared = _overlapping(
            arguments.out.shape, getattr(arguments, operand), end
        )
        wrong = arguments.replaced(out=out, **{operand: shared})
    with pytest.raises(TypeError, match=SHARED_REFUSAL):
        wrong.invoke(_cfdcore.limited_face_values)


# ---------------------------------------------------------------------------
# The implicit solve (prompt 88): ``implicit_step`` against scalar_scheme's.
#
# A Jacobi solve copies the reference's order everywhere, so every output is
# compared bit for bit: the field, the sweeps, converged, the leftover, and the
# number of times the leftover was summed (the reference's ``_Acceptance._form``
# calls, counted). A CG solve's dot products are the reference's OpenBLAS ddot
# and the core's row-blocked sum (ADR-017 G, the second kind), so it is
# compared by prompt 88's rule: the reference's iteration count, and where the
# counts agree the field within 1e-12 of the reference's largest magnitude,
# with differing counts reported apart (in the JUnit record's properties). The
# reference's CG with its ``np.vdot`` replaced by the core's ``row_dot`` must
# then be the core's CG bit for bit, which pins every other operation of the
# core's CG to the reference's order.

IMPLICIT_TOL = 1e-10
IMPLICIT_CAP = 200
# The CG cases' tolerance and cap, as `turbulence.tol` and `max_iter` are in
# configs/clean_room_default.yaml; "tight" cases go to 1e-15, where the
# recursion meets the stop before the true residual does and the solve restarts.
CG_TOL = 1e-10
CG_TIGHT_TOL = 1e-15
CG_CAP = 500
# Prompt 88's rule for a CG solve whose count matches the reference's.
CG_FIELD_RTOL = 1e-12
JACOBI_FIELDS = ["decades", "signed", "plume", "every-face"]
CG_FIELDS = ["positive-decades", "eps-like", "tight"]
STEPS = ["scalar", "per-cell"]
BOUNDS = ["none", "binds", "infinite", "zero"]
HOLDS = ["none", "held"]
CG_STEPS = ["scalar", "per-cell", "per-cell-inf-solid"]
# The CG cases with a leftover bound (review 88 B1): a field, a stop and a bound.
# At the "loose" stop the start residual already meets the max-norm test, so
# the acceptance at the start decides whether the solve ends there; at the
# field's own stop the acceptance at a confirmation does. The "tight" field is
# at its tolerance near rounding, so it is run at that stop only.
CG_BOUNDS = ["binds", "infinite", "zero"]
CG_BOUND_STOPS = ["tol", "loose"]
CG_BOUND_CASES = [
    (field, stop, bound)
    for field in ["positive-decades", "tight"]
    for stop in CG_BOUND_STOPS
    for bound in CG_BOUNDS
    if not (field == "tight" and stop == "loose")
]
CG_BOUND_IDS = ["-".join(case) for case in CG_BOUND_CASES]
# A tolerance that the planted systems' start residuals meet (the reach test
# checks it: the unbounded solve ends at the start).
CG_LOOSE_TOL = 1e3
# No leftover is exactly zero, so a zero bound runs a CG solve to its cap and
# then the fallback's Jacobi solve to its cap; a short cap keeps the case cheap.
CG_ZERO_BOUND_CAP = 30
# One seed per mesh, so the cases of one mesh do not repeat another's draws.
MESH_SEEDS = {name: i for i, name in enumerate(MESHES)}


# The reference divides by a_P in every cell, SOLID ones included, before it
# zeroes them; with an infinite step in a SOLID cell a_P is 0 there and the
# quotient 0 / 0 warns. The value is discarded, so the warning is not a finding.
REFERENCE_DIVIDES_BY_ZERO = pytest.mark.filterwarnings(
    "ignore:invalid value encountered in divide:RuntimeWarning"
)


@functools.cache
def _implicit_scratch(name: str) -> "_cfdcore.ImplicitScratch2":
    """The core's implicit scratch on a mesh (it keeps its grid alive)."""
    return _cfdcore.ImplicitScratch2(_grid(_mesh(name)))


@dataclasses.dataclass(frozen=True, eq=False)
class ImplicitArguments(Call):
    """The arguments of ``implicit_step``, in the reference's order."""

    c_star: np.ndarray
    volume: np.ndarray
    dt: float | np.ndarray
    g_u: np.ndarray
    g_v: np.ndarray
    diagonal: np.ndarray
    solid: np.ndarray
    tol: float
    max_iter: int
    held: np.ndarray | None = None
    method: str = "jacobi"
    leftover_bound: float | None = None

    def reference(self) -> ImplicitResult:
        """The reference's solve on these arguments."""
        return implicit_step(
            self.c_star,
            self.volume,
            self.dt,
            self.g_u,
            self.g_v,
            self.diagonal,
            self.solid,
            self.tol,
            self.max_iter,
            held=self.held,
            method=self.method,
            leftover_bound=self.leftover_bound,
        )

    def core(self, scratch: "_cfdcore.ImplicitScratch2") -> tuple[np.ndarray, object]:
        """The core's solve on these arguments: its field and its result."""
        out = np.full(self.c_star.shape, np.nan)
        result = _cfdcore.implicit_step(
            scratch,
            self.c_star,
            self.volume,
            self.dt,
            self.g_u,
            self.g_v,
            self.diagonal,
            self.solid,
            self.tol,
            self.max_iter,
            out,
            held=self.held,
            method=self.method,
            leftover_bound=self.leftover_bound,
        )
        return out, result


def _conductances(
    mesh: Mesh,
    solid: np.ndarray,
    rng: np.random.Generator,
    scale: np.ndarray,
    every_face: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """``G_f = Gamma_f A_f / d_f`` on the faces between two non-SOLID cells, as the solvers form them.

    ``Gamma_f`` is ``scale`` (one value per cell, the face taking its west or
    south cell's) times a random factor over two decades; every domain face
    and every face beside a SOLID cell has no conductance. With
    ``every_face`` every face has one, the first and the last of each row and
    column included: no caller hands that in, but the reference's arithmetic
    serves it, and it is where a SOLID cell's value of C* and the zero read
    beyond each end of a line reach a sum (a face with no conductance
    multiplies that value by zero).
    """
    ny, nx = solid.shape
    inner_u = np.zeros((ny, nx + 1), dtype=bool)
    inner_u[:, 1:-1] = ~solid[:, :-1] & ~solid[:, 1:]
    inner_v = np.zeros((ny + 1, nx), dtype=bool)
    inner_v[1:-1, :] = ~solid[:-1, :] & ~solid[1:, :]
    if every_face:
        inner_u[:, :] = True
        inner_v[:, :] = True
    gamma_u = np.zeros((ny, nx + 1))
    gamma_u[:, 1:] = scale
    gamma_v = np.zeros((ny + 1, nx))
    gamma_v[1:, :] = scale
    if every_face:
        # The first face of a line has no west or south cell to take a value
        # from, so it takes the east or north cell's.
        gamma_u[:, 0] = scale[:, 0]
        gamma_v[0, :] = scale[0, :]
    gamma_u *= 10.0 ** rng.uniform(-2.0, 0.0, gamma_u.shape)
    gamma_v *= 10.0 ** rng.uniform(-2.0, 0.0, gamma_v.shape)
    area_u = np.broadcast_to(mesh.dy_cell[:, None], (ny, nx + 1))
    area_v = np.broadcast_to(mesh.dx_cell[None, :], (ny + 1, nx))
    g_u = np.where(inner_u, gamma_u * area_u / mesh.dx_face[None, :], 0.0)
    g_v = np.where(inner_v, gamma_v * area_v / mesh.dy_face[:, None], 0.0)
    return g_u, g_v


def _plume(mesh: Mesh, solid: np.ndarray) -> np.ndarray:
    """A release's plume: 1e6 at its centre, falling to underflow and zero far from it."""
    xc, yc = np.meshgrid(mesh.xc, mesh.yc)
    x0, y0 = mesh.xc[len(mesh.xc) // 3], mesh.yc[len(mesh.yc) // 2]
    width = 0.04 * (mesh.x[-1] - mesh.x[0])
    field = 1e6 * np.exp(-(((xc - x0) ** 2 + (yc - y0) ** 2) / width**2))
    return np.where(solid, 0.0, field)


def _jacobi_arguments(
    name: str, field: str, step: str, hold: str, rng: np.random.Generator
) -> ImplicitArguments:
    """A planted transport-like system: a class's field, its diffusion and its deposition sink."""
    mesh = _mesh(name)
    solid = mesh.cell_type == SOLID
    volume = np.outer(mesh.dy_cell, mesh.dx_cell)
    shape = solid.shape
    dt_scalar = 0.05
    # dt G / V from about 1e-6 to 0.5, the transport solver's range and below
    # (ADR-016 measured kappa up to 0.42 on 320x120).
    scale = (volume / dt_scalar) * 10.0 ** rng.uniform(-4.0, -0.3, shape)
    scale *= mesh.dx_face[1:][None, :] / mesh.dy_cell[:, None]
    g_u, g_v = _conductances(mesh, solid, rng, scale, every_face=field == "every-face")
    sink = np.where(
        (rng.uniform(size=shape) < 0.3) & ~solid,
        (volume / dt_scalar) * 10.0 ** rng.uniform(-6.0, -1.0, shape),
        0.0,
    )
    if field == "decades":
        c_star = 10.0 ** rng.uniform(-12.0, 4.0, shape)
        c_star[rng.uniform(size=shape) < 0.1] = 0.0
    elif field == "signed":
        c_star = _values(rng, shape)
    elif field == "plume":
        c_star = _plume(mesh, solid)
    else:
        c_star = 10.0 ** rng.uniform(-12.0, 4.0, shape)
    if field != "every-face":
        c_star = np.where(solid, 0.0, c_star)
    else:
        # Jacobi's first residual reads C* as given, SOLID cells included.
        c_star = np.where(solid, 5.0, c_star)
    dt: float | np.ndarray = dt_scalar
    if step == "per-cell":
        dt = dt_scalar * 10.0 ** rng.uniform(-0.5, 0.5, shape)
        # Not read in a SOLID cell: an infinity there must not reach the sum.
        dt[solid] = np.inf
    held = None
    if hold == "held":
        held = (rng.uniform(size=shape) < 0.05) & ~solid
    return ImplicitArguments(
        c_star=np.ascontiguousarray(c_star),
        volume=volume,
        dt=dt,
        g_u=g_u,
        g_v=g_v,
        diagonal=sink,
        solid=solid,
        tol=IMPLICIT_TOL,
        max_iter=IMPLICIT_CAP,
        held=held,
    )


# The planted CG systems' ranges, as powers of ten: the decay rate eps / k (or
# eps / k times C_2 for eps), the pseudo-time step, nu_t, and C*. The "tight"
# system has a weak sink and a long step, so it is conditioned far worse, and
# at CG_TIGHT_TOL its recursion meets the stop before its true residual does
# (results/builder88/tight.py measured the ranges that restart).
CG_RANGES = {
    "positive-decades": {
        "rate": (-2.0, 2.0),
        "dt": (-3.0, -1.0),
        "nu_t": (-5.0, -2.0),
        "c": (-6.0, 2.0),
    },
    "eps-like": {
        "rate": (-1.0, 3.0),
        "dt": (-3.0, -1.0),
        "nu_t": (-5.0, -2.0),
        "c": (-4.0, 4.0),
    },
    "tight": {
        "rate": (-6.0, -2.0),
        "dt": (-1.0, 1.0),
        "nu_t": (-3.0, -1.0),
        "c": (-6.0, 2.0),
    },
}


def _cg_arguments(
    name: str, field: str, step: str, hold: str, rng: np.random.Generator
) -> ImplicitArguments:
    """A planted k-or-eps-like system: positive in every cell it solves for, a per-cell step."""
    mesh = _mesh(name)
    solid = mesh.cell_type == SOLID
    volume = np.outer(mesh.dy_cell, mesh.dx_cell)
    shape = solid.shape
    ranges = CG_RANGES[field]
    # The local pseudo-time step of ADR-012 C, the largest finite value in SOLID cells.
    dt_cells = 10.0 ** rng.uniform(*ranges["dt"], shape)
    dt_cells[solid] = dt_cells[~solid].max()
    if step == "scalar":
        dt: float | np.ndarray = float(np.median(dt_cells))
    elif step == "per-cell":
        dt = dt_cells
    else:
        dt = np.where(solid, np.inf, dt_cells)
    nu_t = 10.0 ** rng.uniform(*ranges["nu_t"], shape)
    g_u, g_v = _conductances(mesh, solid, rng, nu_t)
    c_star = np.where(solid, 0.0, 10.0 ** rng.uniform(*ranges["c"], shape))
    sink = np.where(solid, 0.0, volume * 10.0 ** rng.uniform(*ranges["rate"], shape))
    held = None
    if hold == "held":
        held = (rng.uniform(size=shape) < 0.05) & ~solid
    return ImplicitArguments(
        c_star=c_star,
        volume=volume,
        dt=dt,
        g_u=g_u,
        g_v=g_v,
        diagonal=sink,
        solid=solid,
        tol=CG_TIGHT_TOL if field == "tight" else CG_TOL,
        max_iter=CG_CAP,
        held=held,
        method="cg",
    )


@dataclasses.dataclass
class _Counts:
    """How often the reference summed a leftover and formed a neighbour sum."""

    forms: int = 0
    neighbour_sums: int = 0


@contextlib.contextmanager
def _counting() -> Iterator[_Counts]:
    """Count the reference's leftover sums and neighbour sums while the block runs."""
    counts = _Counts()
    form = scalar_scheme._Acceptance._form
    neighbour_sum = scalar_scheme._neighbour_sum

    def counted_form(self: object, residual: np.ndarray) -> float:
        counts.forms += 1
        return form(self, residual)

    def counted_neighbour_sum(*args: np.ndarray) -> np.ndarray:
        counts.neighbour_sums += 1
        return neighbour_sum(*args)

    scalar_scheme._Acceptance._form = counted_form
    scalar_scheme._neighbour_sum = counted_neighbour_sum
    try:
        yield counts
    finally:
        scalar_scheme._Acceptance._form = form
        scalar_scheme._neighbour_sum = neighbour_sum


def _reference_restarts(result: ImplicitResult, counts: _Counts) -> int:
    """The reference CG's restarts, from its neighbour sums.

    Each CG iteration forms one product; the start forms one true residual and
    every confirmation another, the last confirmation (a stop or the cap) ending
    the solve and every other one restarting it; a fallback's Jacobi forms one
    residual per sweep and one more.
    """
    if result.iterations == 0:
        return 0
    fallback = result.fallback_sweeps + 1 if result.fallbacks else 0
    return counts.neighbour_sums - result.iterations - 2 - fallback


def _reference_counted(
    arguments: ImplicitArguments,
) -> tuple[ImplicitResult, _Counts]:
    """The reference's solve with its counts."""
    with _counting() as counts:
        result = arguments.reference()
    return result, counts


def _assert_bit_for_bit(
    arguments: ImplicitArguments,
    reference: ImplicitResult,
    counts: _Counts,
    out: np.ndarray,
    core: object,
) -> None:
    """Every output of the core's solve equals the reference's, the leftover's sums counted."""
    assert _same_bits(out, reference.field)
    assert core.method == reference.method
    assert core.iterations == reference.iterations
    assert core.converged == reference.converged
    assert core.fallbacks == reference.fallbacks
    assert core.fallback_sweeps == reference.fallback_sweeps
    assert _bits(core.leftover) == _bits(reference.leftover)
    assert core.leftover_sums == counts.forms


def _bound(arguments: ImplicitArguments, kind: str) -> float | None:
    """The leftover bound of a case: none, one that binds, an infinite one or zero.

    The binding bound is an eighth of the reference's leftover without a
    bound, so the stop takes more sweeps wherever that leftover is not zero.
    """
    if kind == "none":
        return None
    if kind == "infinite":
        return np.inf
    if kind == "zero":
        return 0.0
    return abs(arguments.reference().leftover) / 8.0


@REFERENCE_DIVIDES_BY_ZERO
@pytest.mark.parametrize("hold", HOLDS)
@pytest.mark.parametrize("bound", BOUNDS)
@pytest.mark.parametrize("step", STEPS)
@pytest.mark.parametrize("field", JACOBI_FIELDS)
@pytest.mark.parametrize("name", list(MESHES))
def test_implicit_jacobi_equals_the_reference_bit_for_bit(
    name: str,
    field: str,
    step: str,
    bound: str,
    hold: str,
    record_property: Callable[[str, object], None],
) -> None:
    """A Jacobi solve: field, sweeps, converged, leftover and its sums, bit for bit (prediction (a))."""
    rng = np.random.default_rng(
        [
            SEED,
            MESH_SEEDS[name],
            JACOBI_FIELDS.index(field),
            STEPS.index(step),
            HOLDS.index(hold),
        ]
    )
    arguments = _jacobi_arguments(name, field, step, hold, rng)
    unbound = arguments.reference()
    arguments = arguments.replaced(leftover_bound=_bound(arguments, bound))
    reference, counts = _reference_counted(arguments)
    out, core = arguments.core(_implicit_scratch(name))
    record_property("sweeps", reference.iterations)
    record_property("sweeps_unbound", unbound.iterations)
    record_property("converged", reference.converged)
    record_property("leftover", reference.leftover)
    record_property("leftover_sums", counts.forms)
    _assert_bit_for_bit(arguments, reference, counts, out, core)


@REFERENCE_DIVIDES_BY_ZERO
def test_the_planted_jacobi_inputs_reach_every_rule() -> None:
    """The planted cases reach what the bit-for-bit test claims to pin.

    On 80x30: a binding bound adds sweeps; a zero bound runs to the cap; a held
    cell keeps C* and changes its neighbours; with conductances on every face,
    the first and last of every line included, C*'s SOLID values change the
    result, since Jacobi's first residual reads them; and an infinite per-cell
    step in the SOLID cells leaves the leftover finite.
    """
    name = "80x30"
    rng = np.random.default_rng([SEED, 88])
    base = _jacobi_arguments(name, "decades", "scalar", "none", rng)
    free = base.reference()
    bound = base.replaced(leftover_bound=abs(free.leftover) / 8.0).reference()
    assert free.converged and bound.converged and bound.iterations > free.iterations
    zero = base.replaced(leftover_bound=0.0).reference()
    assert zero.iterations == IMPLICIT_CAP and not zero.converged
    held = base.replaced(
        held=(rng.uniform(size=base.c_star.shape) < 0.05) & ~base.solid
    )
    held_result = held.reference()
    assert _same_bits(held_result.field[held.held], base.c_star[held.held])
    assert not _same_bits(held_result.field, free.field)
    every = _jacobi_arguments(name, "every-face", "scalar", "none", rng)
    # The first and the last face of every row and column carry conductance, so
    # the zero read beyond each end of a line is multiplied by something.
    assert (every.g_u[:, 0] > 0).all() and (every.g_u[:, -1] > 0).all()
    assert (every.g_v[0, :] > 0).all() and (every.g_v[-1, :] > 0).all()
    zeroed = every.replaced(c_star=np.where(every.solid, 0.0, every.c_star)).reference()
    assert not _same_bits(every.reference().field, zeroed.field)
    per_cell = _jacobi_arguments(name, "decades", "per-cell", "none", rng)
    assert np.isinf(per_cell.dt[per_cell.solid]).all()
    assert np.isfinite(per_cell.reference().leftover)


@REFERENCE_DIVIDES_BY_ZERO
@pytest.mark.parametrize("hold", HOLDS)
@pytest.mark.parametrize("step", CG_STEPS)
@pytest.mark.parametrize("field", CG_FIELDS)
@pytest.mark.parametrize("name", list(MESHES))
def test_implicit_cg_meets_prompt_88s_rule(
    name: str,
    field: str,
    step: str,
    hold: str,
    record_property: Callable[[str, object], None],
) -> None:
    """A CG solve: the reference's count, and where it agrees the field within 1e-12."""
    rng = np.random.default_rng(
        [
            SEED,
            MESH_SEEDS[name],
            10 + CG_FIELDS.index(field),
            CG_STEPS.index(step),
            HOLDS.index(hold),
        ]
    )
    arguments = _cg_arguments(name, field, step, hold, rng)
    reference = arguments.reference()
    out, core = arguments.core(_implicit_scratch(name))
    largest = float(np.abs(reference.field).max())
    difference = float(np.abs(out - reference.field).max()) / largest
    record_property("reference_iterations", reference.iterations)
    record_property("core_iterations", core.iterations)
    record_property("relative_difference", difference)
    record_property("reference_fallbacks", reference.fallbacks)
    record_property("core_fallbacks", core.fallbacks)
    record_property("restarts", core.restarts)
    if core.iterations == reference.iterations:
        assert difference <= CG_FIELD_RTOL


class _NumpyWithTheCoresDot:
    """NumPy, with ``vdot`` the core's row-blocked dot product (``_cfdcore.row_dot``)."""

    def __getattr__(self, name: str) -> object:
        return getattr(np, name)

    @staticmethod
    def vdot(a: np.ndarray, b: np.ndarray) -> float:
        """The core's dot product of two [ny, nx] arrays, a block per row."""
        return _cfdcore.row_dot(a, b)


@REFERENCE_DIVIDES_BY_ZERO
@pytest.mark.parametrize("hold", HOLDS)
@pytest.mark.parametrize("step", CG_STEPS)
@pytest.mark.parametrize("field", CG_FIELDS)
@pytest.mark.parametrize("name", list(MESHES))
def test_implicit_cg_with_the_cores_dot_products_is_bit_for_bit(
    name: str,
    field: str,
    step: str,
    hold: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The reference's CG with the core's dot products is the core's CG bit for bit.

    Everything but the two dot products is elementwise and copied in the
    reference's order, so with the same dot products the two must agree in
    every bit: the field, the count, converged, the fallback, the leftover and
    its sums, and the restarts.
    """
    rng = np.random.default_rng(
        [
            SEED,
            MESH_SEEDS[name],
            10 + CG_FIELDS.index(field),
            CG_STEPS.index(step),
            HOLDS.index(hold),
        ]
    )
    arguments = _cg_arguments(name, field, step, hold, rng)
    monkeypatch.setattr(scalar_scheme, "np", _NumpyWithTheCoresDot())
    reference, counts = _reference_counted(arguments)
    out, core = arguments.core(_implicit_scratch(name))
    _assert_bit_for_bit(arguments, reference, counts, out, core)
    assert core.restarts == _reference_restarts(reference, counts)


@REFERENCE_DIVIDES_BY_ZERO
def test_the_planted_cg_inputs_reach_every_rule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The tight cases restart, and the held cells keep C*, so the bit-for-bit test pins both."""
    monkeypatch.setattr(scalar_scheme, "np", _NumpyWithTheCoresDot())
    restarted = 0
    for name in MESHES:
        for step in CG_STEPS:
            rng = np.random.default_rng(
                [
                    SEED,
                    MESH_SEEDS[name],
                    10 + CG_FIELDS.index("tight"),
                    CG_STEPS.index(step),
                    1,
                ]
            )
            arguments = _cg_arguments(name, "tight", step, "held", rng)
            result, counts = _reference_counted(arguments)
            restarted += _reference_restarts(result, counts) > 0
            assert _same_bits(
                result.field[arguments.held], arguments.c_star[arguments.held]
            )
    assert restarted >= 3


def _isolated_zero(arguments: ImplicitArguments) -> ImplicitArguments:
    """The arguments with one non-SOLID cell cut off: no conductance, no sink, C* zero.

    CG leaves that cell at zero, which is not positive, so the solve falls back.
    """
    solid = arguments.solid
    j, i = np.argwhere(~solid)[len(np.argwhere(~solid)) // 2]
    g_u, g_v = arguments.g_u.copy(), arguments.g_v.copy()
    g_u[j, i] = g_u[j, i + 1] = 0.0
    g_v[j, i] = g_v[j + 1, i] = 0.0
    sink = arguments.diagonal.copy()
    sink[j, i] = 0.0
    c_star = arguments.c_star.copy()
    c_star[j, i] = 0.0
    held = None if arguments.held is None else arguments.held.copy()
    if held is not None:
        held[j, i] = False
    return arguments.replaced(g_u=g_u, g_v=g_v, diagonal=sink, c_star=c_star, held=held)


@REFERENCE_DIVIDES_BY_ZERO
@pytest.mark.parametrize("cause", ["cap", "non-positive"])
@pytest.mark.parametrize("name", list(MESHES))
def test_implicit_cg_fallback_is_the_jacobi_solve_bit_for_bit(
    name: str, cause: str, record_property: Callable[[str, object], None]
) -> None:
    """A CG solve at its cap, or leaving a zero, falls back to Jacobi from C* as the reference does.

    The fallback's Jacobi solve does not read the CG iterate, so its field,
    sweeps, converged flag and leftover are compared bit for bit even though
    the CG before it is compared by the rule.
    """
    rng = np.random.default_rng([SEED, MESH_SEEDS[name], 30])
    arguments = _cg_arguments(name, "positive-decades", "per-cell", "held", rng)
    if cause == "cap":
        # One iteration at the tight tolerance: no planted system converges in it.
        arguments = arguments.replaced(max_iter=1, tol=CG_TIGHT_TOL)
    else:
        arguments = _isolated_zero(arguments)
    reference, counts = _reference_counted(arguments)
    out, core = arguments.core(_implicit_scratch(name))
    record_property("reference_iterations", reference.iterations)
    record_property("core_iterations", core.iterations)
    assert reference.fallbacks == core.fallbacks == 1
    assert _same_bits(out, reference.field)
    assert core.fallback_sweeps == reference.fallback_sweeps
    assert core.converged == reference.converged
    assert _bits(core.leftover) == _bits(reference.leftover)
    if cause == "cap":
        assert core.iterations == reference.iterations == 1
        assert core.leftover_sums == counts.forms
    # The fallback starts from C*, not from the CG iterate: a Jacobi solve alone
    # gives the same field.
    jacobi = arguments.replaced(method="jacobi").reference()
    assert _same_bits(out, jacobi.field)


def _cg_bound_arguments(
    name: str, field: str, stop: str, bound: str
) -> ImplicitArguments:
    """A CG system with its leftover bound: the stop, the cap and the bound by case.

    The three bounds of a field and a stop share one system, so that they can
    be compared. The binding bound is an eighth of the reference's leftover
    without one, computed on the arguments' reference as the caller has set it
    up, the dot product included.
    """
    rng = np.random.default_rng(
        [
            SEED,
            MESH_SEEDS[name],
            20 + CG_FIELDS.index(field),
            CG_BOUND_STOPS.index(stop),
        ]
    )
    arguments = _cg_arguments(name, field, "per-cell", "none", rng)
    if stop == "loose":
        arguments = arguments.replaced(tol=CG_LOOSE_TOL)
    if bound == "zero":
        arguments = arguments.replaced(max_iter=CG_ZERO_BOUND_CAP)
    return arguments.replaced(leftover_bound=_bound(arguments, bound))


@REFERENCE_DIVIDES_BY_ZERO
@pytest.mark.parametrize("case", CG_BOUND_CASES, ids=CG_BOUND_IDS)
@pytest.mark.parametrize("name", list(MESHES))
def test_implicit_cg_with_a_leftover_bound_meets_prompt_88s_rule(
    name: str,
    case: tuple[str, str, str],
    record_property: Callable[[str, object], None],
) -> None:
    """A CG solve with a bound: the reference's count, the field within 1e-12, the bound kept.

    The rule is the unbounded CG cases': where the counts agree the field is
    within 1e-12 of the reference's largest magnitude, and differing counts are
    recorded. A bound adds one more thing the reference does not decide: the
    leftover of a field the solve accepts is inside the bound (ADR-016), so a
    converged result, the core's or the reference's, must show it. The bound's
    test reads a cancelling sum, so counts can differ at ordinary tolerances
    here, where the unbounded cases' do not.
    """
    field, stop, kind = case
    arguments = _cg_bound_arguments(name, field, stop, kind)
    bound = arguments.leftover_bound
    reference = arguments.reference()
    out, core = arguments.core(_implicit_scratch(name))
    largest = float(np.abs(reference.field).max())
    difference = float(np.abs(out - reference.field).max()) / largest
    record_property("reference_iterations", reference.iterations)
    record_property("core_iterations", core.iterations)
    record_property("relative_difference", difference)
    record_property("reference_fallbacks", reference.fallbacks)
    record_property("core_fallbacks", core.fallbacks)
    record_property("restarts", core.restarts)
    record_property("converged", core.converged)
    if core.iterations == reference.iterations:
        assert difference <= CG_FIELD_RTOL
    for result in (reference, core):
        if result.converged:
            assert abs(result.leftover) <= bound


@REFERENCE_DIVIDES_BY_ZERO
@pytest.mark.parametrize("case", CG_BOUND_CASES, ids=CG_BOUND_IDS)
@pytest.mark.parametrize("name", list(MESHES))
def test_implicit_cg_with_a_leftover_bound_and_the_cores_dot_products_is_bit_for_bit(
    name: str,
    case: tuple[str, str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With the core's dot products the reference's bounded CG is the core's bit for bit.

    The acceptance at the start, at each confirmation and at the cap, the
    restarts on the bound's account and the fallback's own acceptance all show
    in the field, the counts, the leftover and the number of times it was
    summed.
    """
    field, stop, kind = case
    monkeypatch.setattr(scalar_scheme, "np", _NumpyWithTheCoresDot())
    arguments = _cg_bound_arguments(name, field, stop, kind)
    reference, counts = _reference_counted(arguments)
    out, core = arguments.core(_implicit_scratch(name))
    _assert_bit_for_bit(arguments, reference, counts, out, core)
    assert core.restarts == _reference_restarts(reference, counts)


@REFERENCE_DIVIDES_BY_ZERO
@pytest.mark.parametrize("name", ["80x30", "160x60-stretched"])
def test_the_planted_cg_bound_inputs_reach_every_rule(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each bound does something to the planted CG solve: the bit-for-bit test pins what it reaches.

    A binding bound adds iterations and restarts to the unbounded solve, whose
    confirmation it refuses; an infinite bound changes neither; a zero bound
    runs the solve to its cap and the fallback after it, converged nowhere; at
    the loose stop the unbounded solve ends at the start and a binding bound
    refuses that, so the solve goes on.
    """
    monkeypatch.setattr(scalar_scheme, "np", _NumpyWithTheCoresDot())

    def solved(field: str, stop: str, kind: str) -> tuple[ImplicitResult, int]:
        arguments = _cg_bound_arguments(name, field, stop, kind)
        result, counts = _reference_counted(arguments)
        return result, _reference_restarts(result, counts)

    free, free_restarts = solved("positive-decades", "tol", "infinite")
    binds, binds_restarts = solved("positive-decades", "tol", "binds")
    assert free.converged and binds.converged
    assert binds.iterations > free.iterations
    assert binds_restarts > free_restarts
    zero, _ = solved("positive-decades", "tol", "zero")
    assert not zero.converged and zero.fallbacks == 1
    assert zero.iterations == zero.fallback_sweeps == CG_ZERO_BOUND_CAP
    loose, _ = solved("positive-decades", "loose", "infinite")
    refused, _ = solved("positive-decades", "loose", "binds")
    assert loose.iterations == 0 and loose.converged
    assert refused.iterations > 0 and refused.converged
    tight, _ = solved("tight", "tol", "binds")
    assert tight.iterations > 0


@REFERENCE_DIVIDES_BY_ZERO
@pytest.mark.parametrize("name", list(MESHES))
def test_implicit_cg_positivity_is_read_in_the_cells_it_solves_for_only(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A held cell at zero is no reason to fall back: the test is over the unknowns only.

    CG never changes a held cell, so it stays at C*; the reference's test,
    ``c[~fixed] > 0``, leaves it out, and a core that read it would fall back
    from a solve that converged.
    """
    rng = np.random.default_rng([SEED, MESH_SEEDS[name], 31])
    arguments = _cg_arguments(name, "positive-decades", "per-cell", "held", rng)
    assert arguments.held.any()
    arguments = arguments.replaced(
        c_star=np.where(arguments.held, 0.0, arguments.c_star)
    )
    monkeypatch.setattr(scalar_scheme, "np", _NumpyWithTheCoresDot())
    reference, counts = _reference_counted(arguments)
    out, core = arguments.core(_implicit_scratch(name))
    assert reference.fallbacks == core.fallbacks == 0 and reference.converged
    assert _same_bits(out[arguments.held], np.zeros(int(arguments.held.sum())))
    _assert_bit_for_bit(arguments, reference, counts, out, core)


@REFERENCE_DIVIDES_BY_ZERO
@pytest.mark.parametrize("name", list(MESHES))
def test_implicit_cg_iterations_under_a_fallback_are_the_cgs_own(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fallback keeps the CG's iteration count and records the Jacobi sweeps apart.

    One cell left at zero fails the positivity test after the CG converged, so
    Jacobi solves from C*. The two counts must differ here for the case to tell
    them apart, and with the core's dot products the CG's count is the
    reference's.
    """
    rng = np.random.default_rng([SEED, MESH_SEEDS[name], 32])
    # The tight field is conditioned far worse than the others, so at the
    # ordinary tolerance Jacobi needs many more sweeps than CG iterations (at
    # its own 1e-15 both reach the cap).
    arguments = _isolated_zero(
        _cg_arguments(name, "tight", "per-cell", "held", rng)
    ).replaced(tol=CG_TOL)
    monkeypatch.setattr(scalar_scheme, "np", _NumpyWithTheCoresDot())
    reference, counts = _reference_counted(arguments)
    out, core = arguments.core(_implicit_scratch(name))
    assert reference.fallbacks == core.fallbacks == 1
    assert reference.iterations != reference.fallback_sweeps
    _assert_bit_for_bit(arguments, reference, counts, out, core)
    assert core.iterations != core.fallback_sweeps
    assert core.restarts == _reference_restarts(reference, counts)


@REFERENCE_DIVIDES_BY_ZERO
@pytest.mark.parametrize("step", STEPS)
@pytest.mark.parametrize("method", ["jacobi", "cg"])
@pytest.mark.parametrize("name", list(MESHES))
def test_implicit_a_zero_field_meets_the_stop_and_a_zero_bound_with_equality(
    name: str, method: str, step: str
) -> None:
    """C* all zero with a zero bound: the stop, 0 <= 0, and the bound, 0 <= 0, both hold.

    The state of every class's solve before a release: b is zero, so the stop is
    zero and so is the residual, and the leftover is zero against ADR-016's bound
    of zero. A strict comparison in either place makes the solve run on. Under
    CG the all-zero iterate is not positive, so the solve falls back to Jacobi,
    which meets both at once as well.
    """
    rng = np.random.default_rng([SEED, MESH_SEEDS[name], 80, STEPS.index(step)])
    if method == "jacobi":
        arguments = _jacobi_arguments(name, "decades", step, "none", rng)
    else:
        arguments = _cg_arguments(name, "positive-decades", step, "none", rng)
    arguments = arguments.replaced(
        c_star=np.zeros_like(arguments.c_star), leftover_bound=0.0
    )
    reference, counts = _reference_counted(arguments)
    out, core = arguments.core(_implicit_scratch(name))
    assert reference.iterations == 0 and reference.converged
    assert reference.fallback_sweeps == 0
    assert reference.fallbacks == (1 if method == "cg" else 0)
    _assert_bit_for_bit(arguments, reference, counts, out, core)


@pytest.mark.parametrize("method", ["jacobi", "cg"])
@pytest.mark.parametrize("zero", [0.0, -0.0])
@pytest.mark.parametrize("name", ["80x30", "160x60-stretched"])
def test_implicit_nothing_to_solve_returns_c_star(
    name: str, zero: float, method: str
) -> None:
    """No conductance and no sink (zeros of either sign): C* as given, with no solve."""
    rng = np.random.default_rng([SEED, MESH_SEEDS[name], 40])
    arguments = _jacobi_arguments(name, "every-face", "scalar", "none", rng)
    arguments = arguments.replaced(
        g_u=np.full_like(arguments.g_u, zero),
        g_v=np.full_like(arguments.g_v, zero),
        diagonal=np.full_like(arguments.diagonal, zero),
        method=method,
    )
    reference, counts = _reference_counted(arguments)
    out, core = arguments.core(_implicit_scratch(name))
    assert reference.field is arguments.c_star
    _assert_bit_for_bit(arguments, reference, counts, out, core)
    assert core.iterations == 0 and core.leftover_sums == 0


@REFERENCE_DIVIDES_BY_ZERO
@pytest.mark.parametrize("step", STEPS)
@pytest.mark.parametrize("name", list(MESHES))
def test_implicit_system_equals_the_references_set_up(
    name: str, step: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """a_P, b and the stop are the values the reference hands its Jacobi solve, bit for bit."""
    rng = np.random.default_rng([SEED, MESH_SEEDS[name], 50, STEPS.index(step)])
    arguments = _jacobi_arguments(name, "signed", step, "none", rng)
    seen: dict[str, object] = {}
    original = scalar_scheme._jacobi

    def recording(
        c_star: np.ndarray, a_p: np.ndarray, b: np.ndarray, *rest: object
    ) -> object:
        seen.update(a_p=a_p, b=b, stop=rest[4])
        return original(c_star, a_p, b, *rest)

    monkeypatch.setattr(scalar_scheme, "_jacobi", recording)
    arguments.reference()
    scratch = _implicit_scratch(name)
    diffuses, stop = _cfdcore.implicit_system(
        scratch,
        arguments.c_star,
        arguments.volume,
        arguments.dt,
        arguments.g_u,
        arguments.g_v,
        arguments.diagonal,
        arguments.tol,
    )
    assert diffuses
    assert _bits(stop) == _bits(seen["stop"])
    assert _same_bits(scratch.diagonal(), np.ravel(seen["a_p"]))
    assert _same_bits(scratch.rhs(), np.ravel(seen["b"]))


@pytest.mark.parametrize("name", ["80x30", "160x60-stretched"])
def test_implicit_stop_reads_b_in_every_cell(name: str) -> None:
    """The stop is tol times the largest |b| over every cell, SOLID cells included.

    C* is a thousand times its largest value elsewhere in the SOLID cells,
    where the callers keep it zero. Under the conductances the solvers form, a
    SOLID value reaches no neighbour, so it changes the stop alone: the
    reference stops sooner than with those cells at zero, and the core with
    it, bit for bit.
    """
    rng = np.random.default_rng([SEED, MESH_SEEDS[name], 55])
    quiet = _jacobi_arguments(name, "decades", "scalar", "none", rng)
    loud = quiet.replaced(
        c_star=np.where(quiet.solid, 1e3 * quiet.c_star.max(), quiet.c_star)
    )
    assert loud.reference().iterations < quiet.reference().iterations
    reference, counts = _reference_counted(loud)
    out, core = loud.core(_implicit_scratch(name))
    _assert_bit_for_bit(loud, reference, counts, out, core)


def test_row_dot_is_the_row_partition_and_differs_from_vdot_in_rounding_only() -> None:
    """row_dot sums each row pairwise and adds the rows to 0.0 in order; np.vdot is another order."""
    rng = np.random.default_rng([SEED, 60])
    departs = 0
    for shape in SHAPES:
        a, b = _values(rng, shape), _values(rng, shape)
        expected = 0.0
        for row_a, row_b in zip(a, b, strict=True):
            expected += _cfdcore.pairwise_sum(np.ascontiguousarray(row_a * row_b))
        assert _bits(_cfdcore.row_dot(a, b)) == _bits(expected), shape
        magnitude = float(np.sum(np.abs(a * b)))
        assert abs(_cfdcore.row_dot(a, b) - float(np.vdot(a, b))) <= 1e-13 * magnitude
        departs += _bits(_cfdcore.row_dot(a, b)) != _bits(float(np.vdot(a, b)))
    # The comparison with ddot is the second kind's: the bits are not np.vdot's.
    assert departs > 0


@pytest.mark.parametrize("shape", [(3, 0), (0, 4), (0, 0)])
def test_row_dot_of_arrays_with_no_terms_is_zero(shape: tuple[int, int]) -> None:
    """An array with no rows or no columns has no terms: 0.0, as np.vdot gives.

    blocked_dot refuses a block of zero, and an array with no columns has a row
    length of zero, so the binding hands it a block of 1, which is never read.
    """
    a = np.zeros(shape)
    assert float(np.vdot(a, a)) == 0.0
    assert _bits(_cfdcore.row_dot(a, a)) == _bits(0.0)


@pytest.mark.parametrize(
    "a, b, message",
    [
        pytest.param(np.ones((3, 4)), np.ones((3, 5)), "differ in shape", id="shapes"),
        pytest.param(
            np.ones((3, 4)), np.ones((3, 4), np.float32), NANOBIND_REFUSAL, id="dtype"
        ),
        pytest.param(
            np.ones((3, 4)), _strided(np.ones((3, 4))), NANOBIND_REFUSAL, id="strided"
        ),
        pytest.param(
            np.ones((3, 4)),
            _unaligned_like(np.ones((3, 4))),
            UNALIGNED_REFUSAL,
            id="unaligned",
        ),
    ],
)
def test_row_dot_refuses_wrong_arrays(
    a: np.ndarray, b: np.ndarray, message: str
) -> None:
    """row_dot refuses arrays of two shapes, another dtype or layout, and an unaligned one."""
    with pytest.raises(TypeError, match=message):
        _cfdcore.row_dot(a, b)


def _small_implicit_arguments() -> ImplicitArguments:
    """A consistent planted system on the 40x15 mesh, for the refusal tests."""
    rng = np.random.default_rng([SEED, 70])
    return _jacobi_arguments("40x15-uniform-open", "decades", "per-cell", "held", rng)


@pytest.mark.parametrize(
    "change",
    [
        pytest.param({"method": "gauss"}, id="method-unknown"),
        pytest.param({"method": "JACOBI"}, id="method-case"),
        pytest.param({"method": None}, id="method-none"),
        pytest.param({"leftover_bound": True}, id="bound-python-bool"),
        pytest.param({"leftover_bound": np.True_}, id="bound-numpy-bool"),
        pytest.param({"leftover_bound": False}, id="bound-false"),
        pytest.param({"leftover_bound": -1e-300}, id="bound-negative"),
        pytest.param({"leftover_bound": np.nan}, id="bound-nan"),
        pytest.param(
            {"method": "gauss", "leftover_bound": True}, id="method-before-bound"
        ),
    ],
)
def test_implicit_step_refuses_what_the_reference_refuses(
    change: dict[str, object],
) -> None:
    """The reference's refusals, with its exception type and message, in its order."""
    arguments = _small_implicit_arguments().replaced(**change)
    with pytest.raises((TypeError, ValueError)) as expected:
        arguments.reference()
    with pytest.raises(type(expected.value)) as raised:
        arguments.core(_implicit_scratch("40x15-uniform-open"))
    assert str(raised.value) == str(expected.value)


def test_implicit_step_accepts_consistent_arguments() -> None:
    """The arguments _small_implicit_arguments builds are accepted, so each refusal is its change's."""
    arguments = _small_implicit_arguments()
    arguments.core(_implicit_scratch("40x15-uniform-open"))
    arguments.replaced(dt=0.05, held=None).core(_implicit_scratch("40x15-uniform-open"))


IMPLICIT_ARRAYS = [
    "c_star",
    "volume",
    "dt",
    "g_u",
    "g_v",
    "diagonal",
    "solid",
    "held",
    "out",
]
IMPLICIT_REFUSALS = [
    Refusal(key, change, message, f"{key.replace('_', '-')}-{label}")
    for key in IMPLICIT_ARRAYS
    for label, change, message in (
        ("fewer-rows", _fewer_rows, "is not \\["),
        ("fewer-columns", _fewer_columns, "is not \\["),
        ("non-contiguous", _strided, NANOBIND_REFUSAL),
        ("fortran-order", _fortran, NANOBIND_REFUSAL),
        ("wrong-dtype", _wrong_dtype, NANOBIND_REFUSAL),
        ("unaligned", _unaligned_like, UNALIGNED_REFUSAL),
    )
    # A bool is one byte, so no bool array is unaligned.
    if not (label == "unaligned" and key in ("solid", "held"))
]


def _implicit_with(
    arguments: ImplicitArguments, key: str, value: np.ndarray
) -> Callable[[], object]:
    """A call of the core's implicit_step with one argument (or out) replaced."""
    scratch = _implicit_scratch("40x15-uniform-open")
    out = np.zeros(arguments.c_star.shape)
    if key == "out":
        out = value
    else:
        arguments = arguments.replaced(**{key: value})
    return lambda: _cfdcore.implicit_step(
        scratch,
        arguments.c_star,
        arguments.volume,
        arguments.dt,
        arguments.g_u,
        arguments.g_v,
        arguments.diagonal,
        arguments.solid,
        arguments.tol,
        arguments.max_iter,
        out,
        held=arguments.held,
        method=arguments.method,
        leftover_bound=arguments.leftover_bound,
    )


@refusals(IMPLICIT_REFUSALS)
def test_implicit_step_refuses_wrong_arrays(
    key: str, change: Callable[[np.ndarray], np.ndarray], message: str
) -> None:
    """A wrong shape, dtype, layout or alignment of any array is a TypeError at the binding."""
    arguments = _small_implicit_arguments()
    original = (
        np.zeros(arguments.c_star.shape) if key == "out" else getattr(arguments, key)
    )
    call = _implicit_with(arguments, key, change(original))
    if key == "dt" and message == NANOBIND_REFUSAL:
        # dt is read as a number or an array: an array it cannot take is named.
        message = "dt is a number or a C-contiguous float64 cell field"
    with pytest.raises(TypeError, match=message):
        call()


@pytest.mark.parametrize("end", ["tail", "head"])
@pytest.mark.parametrize("operand", [k for k in IMPLICIT_ARRAYS if k != "out"])
def test_implicit_step_refuses_an_out_that_overlaps_an_input(
    operand: str, end: str
) -> None:
    """An out sharing memory with one input, at either end, is a TypeError."""
    arguments = _small_implicit_arguments()
    out, shared = _overlapping(arguments.c_star.shape, getattr(arguments, operand), end)
    arguments = arguments.replaced(**{operand: shared})
    call = _implicit_with(arguments, "out", out)
    with pytest.raises(TypeError, match=SHARED_REFUSAL):
        call()


# --- The block walk at every row length (prompt 89) -------------------------
#
# The core's kernels walk a row of cells in blocks of eight and take the cells
# that are left one at a time, the row's end cells apart
# (core/include/cfdcore/lanes.hpp). Every mesh above is a multiple of eight
# cells wide, so each leaves the same number of cells and none is shorter than
# a block. The cases below run the bit-for-bit comparisons on small grids of
# every width from one cell to past two blocks and of one row and more, so a
# block that starts or ends one cell out of place reads a wrong neighbour in
# some case. The grids are built here, not from a configuration: the reference
# takes arrays and an ``Axis``, the core a ``Grid2``.

ROW_WIDTHS = [*range(1, 20), 26, 33]
ROW_HEIGHTS = [1, 2, 3, 5]
ROW_SHAPES = [(ny, nx) for nx in ROW_WIDTHS for ny in ROW_HEIGHTS]
ROW_IDS = [f"{nx}x{ny}" for ny, nx in ROW_SHAPES]
ROW_SOLID_SHARE = 0.2
ROW_CAP = 40
ROW_CG_CAP = 200


@dataclasses.dataclass(frozen=True, eq=False)
class RowGrid:
    """A small stretched grid with SOLID cells: the arrays ``Mesh`` would hold."""

    x: np.ndarray
    xc: np.ndarray
    y: np.ndarray
    yc: np.ndarray
    cell_type: np.ndarray

    @property
    def solid(self) -> np.ndarray:
        """The SOLID mask, [ny, nx]."""
        return self.cell_type == SOLID

    @property
    def volume(self) -> np.ndarray:
        """Cell volumes per metre of depth, [ny, nx]."""
        return np.outer(np.diff(self.y), np.diff(self.x))

    def core(self) -> "_cfdcore.Grid2":
        """The core's grid on these arrays."""
        return _cfdcore.Grid2(self.x, self.xc, self.y, self.yc, self.cell_type, SOLID)

    def axes(self) -> tuple[scalar_scheme.Axis, scalar_scheme.Axis]:
        """The reference's two axes, as ``mesh_axes`` builds them from a mesh."""
        ny, nx = self.cell_type.shape
        return (
            scalar_scheme.Axis(
                nodes=np.concatenate(([self.x[0]], self.xc, [self.x[-1]])),
                faces=self.x[1:-1],
                left=np.arange(1, nx),
                solid_ext=np.pad(self.solid, ((0, 0), (1, 1))),
            ),
            scalar_scheme.Axis(
                nodes=np.concatenate(([self.y[0]], self.yc, [self.y[-1]])),
                faces=self.y[1:-1],
                left=np.arange(1, ny),
                solid_ext=np.ascontiguousarray(np.pad(self.solid, ((1, 1), (0, 0))).T),
            ),
        )


def _row_grid(ny: int, nx: int, rng: np.random.Generator) -> RowGrid:
    """A grid of unequal cells with about a fifth of them SOLID, one cell at least not."""
    x = np.concatenate(([0.0], np.cumsum(rng.uniform(0.5, 1.5, nx))))
    y = np.concatenate(([0.0], np.cumsum(rng.uniform(0.5, 1.5, ny))))
    solid = rng.uniform(size=(ny, nx)) < ROW_SOLID_SHARE
    solid.flat[rng.integers(solid.size)] = False
    return RowGrid(
        x=x,
        xc=0.5 * (x[:-1] + x[1:]),
        y=y,
        yc=0.5 * (y[:-1] + y[1:]),
        cell_type=np.where(solid, SOLID, 0).astype(np.int32),
    )


def _row_system(
    grid: RowGrid, method: str, step: str, hold: str, rng: np.random.Generator
) -> ImplicitArguments:
    """A system with conductance on every face, the first and last of each line included.

    Under Jacobi C* spans sixteen decades with zeros, and is 5 in the SOLID
    cells, which the first residual reads; under CG it is positive in every
    cell it solves for and zero in the SOLID ones, as the callers hand it.
    """
    solid = grid.solid
    shape = solid.shape
    ny, nx = shape
    volume = grid.volume
    dt_scalar = 0.05
    g_u = (volume.mean() / dt_scalar) * 10.0 ** rng.uniform(-3.0, -0.3, (ny, nx + 1))
    g_v = (volume.mean() / dt_scalar) * 10.0 ** rng.uniform(-3.0, -0.3, (ny + 1, nx))
    sink = np.where(
        rng.uniform(size=shape) < 0.3,
        (volume / dt_scalar) * 10.0 ** rng.uniform(-6.0, -1.0, shape),
        0.0,
    )
    if method == "jacobi":
        c_star = 10.0 ** rng.uniform(-12.0, 4.0, shape)
        c_star[rng.uniform(size=shape) < 0.1] = 0.0
        c_star = np.where(solid, 5.0, c_star)
    else:
        c_star = np.where(solid, 0.0, 10.0 ** rng.uniform(-6.0, 2.0, shape))
    dt: float | np.ndarray = dt_scalar
    if step == "per-cell":
        dt = dt_scalar * 10.0 ** rng.uniform(-0.5, 0.5, shape)
        if method == "jacobi":
            # Not read in a SOLID cell: an infinity there must not reach the sum.
            dt[solid] = np.inf
    held = None
    if hold == "held":
        held = (rng.uniform(size=shape) < 0.2) & ~solid
    return ImplicitArguments(
        c_star=np.ascontiguousarray(c_star),
        volume=volume,
        dt=dt,
        g_u=g_u,
        g_v=g_v,
        diagonal=sink,
        solid=solid,
        tol=IMPLICIT_TOL,
        max_iter=ROW_CAP if method == "jacobi" else ROW_CG_CAP,
        held=held,
        method=method,
    )


def test_the_row_widths_leave_every_number_of_cells_after_the_blocks() -> None:
    """The widths cover each count of cells a block walk can leave, and two blocks.

    A stencil's walk takes a row's two end cells apart and the nx - 2 between
    in blocks; the flux's takes the first and the last interior face apart and
    the nx - 3 between; an elementwise walk takes all nx.
    """
    block = 8
    for apart in (0, 2, 3):
        left = {(nx - apart) % block for nx in ROW_WIDTHS if nx >= apart}
        assert left == set(range(block)), apart
        assert any(nx - apart >= 2 * block for nx in ROW_WIDTHS), apart
    assert any(nx < block for nx in ROW_WIDTHS)


@REFERENCE_DIVIDES_BY_ZERO
@pytest.mark.parametrize("hold", HOLDS)
@pytest.mark.parametrize("step", STEPS)
@pytest.mark.parametrize(("ny", "nx"), ROW_SHAPES, ids=ROW_IDS)
def test_implicit_jacobi_is_bit_for_bit_at_every_row_length(
    ny: int, nx: int, step: str, hold: str
) -> None:
    """A Jacobi solve on a grid of any width: every output the reference's bit for bit."""
    rng = np.random.default_rng(
        [SEED, 89, ny, nx, STEPS.index(step), HOLDS.index(hold)]
    )
    grid = _row_grid(ny, nx, rng)
    arguments = _row_system(grid, "jacobi", step, hold, rng)
    reference, counts = _reference_counted(arguments)
    out, core = arguments.core(_cfdcore.ImplicitScratch2(grid.core()))
    _assert_bit_for_bit(arguments, reference, counts, out, core)


@pytest.mark.parametrize("bound", ["binds", "zero"])
@pytest.mark.parametrize(("ny", "nx"), ROW_SHAPES, ids=ROW_IDS)
def test_implicit_jacobi_with_a_bound_is_bit_for_bit_at_every_row_length(
    ny: int, nx: int, bound: str
) -> None:
    """The same with ADR-016's bound: the leftover's sums and the sweeps it adds."""
    rng = np.random.default_rng([SEED, 89, ny, nx, 7, BOUNDS.index(bound)])
    grid = _row_grid(ny, nx, rng)
    arguments = _row_system(grid, "jacobi", "scalar", "none", rng)
    arguments = arguments.replaced(leftover_bound=_bound(arguments, bound))
    reference, counts = _reference_counted(arguments)
    out, core = arguments.core(_cfdcore.ImplicitScratch2(grid.core()))
    _assert_bit_for_bit(arguments, reference, counts, out, core)


@pytest.mark.parametrize("hold", HOLDS)
@pytest.mark.parametrize("step", STEPS)
@pytest.mark.parametrize(("ny", "nx"), ROW_SHAPES, ids=ROW_IDS)
def test_implicit_cg_with_the_cores_dot_products_is_bit_for_bit_at_every_row_length(
    ny: int, nx: int, step: str, hold: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A CG solve on a grid of any width, the reference with the core's dot products."""
    rng = np.random.default_rng(
        [SEED, 89, ny, nx, 10 + STEPS.index(step), HOLDS.index(hold)]
    )
    grid = _row_grid(ny, nx, rng)
    arguments = _row_system(grid, "cg", step, hold, rng)
    monkeypatch.setattr(scalar_scheme, "np", _NumpyWithTheCoresDot())
    reference, counts = _reference_counted(arguments)
    out, core = arguments.core(_cfdcore.ImplicitScratch2(grid.core()))
    _assert_bit_for_bit(arguments, reference, counts, out, core)
    assert core.restarts == _reference_restarts(reference, counts)


@pytest.mark.parametrize("step", STEPS)
@pytest.mark.parametrize(("ny", "nx"), ROW_SHAPES, ids=ROW_IDS)
def test_implicit_system_is_the_references_set_up_at_every_row_length(
    ny: int, nx: int, step: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """a_P, b and the stop on a grid of any width are what the reference hands its solve."""
    rng = np.random.default_rng([SEED, 89, ny, nx, 20 + STEPS.index(step)])
    grid = _row_grid(ny, nx, rng)
    arguments = _row_system(grid, "cg", step, "none", rng)
    seen: dict[str, object] = {}
    original = scalar_scheme._jacobi

    def recording(
        c_star: np.ndarray, a_p: np.ndarray, b: np.ndarray, *rest: object
    ) -> object:
        seen.update(a_p=a_p, b=b, stop=rest[4])
        return original(c_star, a_p, b, *rest)

    monkeypatch.setattr(scalar_scheme, "_jacobi", recording)
    arguments.replaced(method="jacobi", max_iter=0).reference()
    scratch = _cfdcore.ImplicitScratch2(grid.core())
    diffuses, stop = _cfdcore.implicit_system(
        scratch,
        arguments.c_star,
        arguments.volume,
        arguments.dt,
        arguments.g_u,
        arguments.g_v,
        arguments.diagonal,
        arguments.tol,
    )
    assert diffuses
    assert _bits(stop) == _bits(seen["stop"])
    assert _same_bits(scratch.diagonal(), np.ravel(seen["a_p"]))
    assert _same_bits(scratch.rhs(), np.ravel(seen["b"]))


@pytest.mark.parametrize("vents", ["none", "random-mask"])
@pytest.mark.parametrize("upwind", [False, True])
@pytest.mark.parametrize("d", [0, 1])
@pytest.mark.parametrize(("ny", "nx"), ROW_SHAPES, ids=ROW_IDS)
def test_advective_flux_is_bit_for_bit_at_every_row_length(
    ny: int, nx: int, d: int, upwind: bool, vents: str
) -> None:
    """The flux on a grid of any width and height, both directions, bit for bit."""
    rng = np.random.default_rng(
        [SEED, 89, ny, nx, 30 + d, int(upwind), int(vents == "none")]
    )
    grid = _row_grid(ny, nx, rng)
    shape = (ny, nx + 1) if d == 0 else (ny + 1, nx)
    # Random values, or a few levels, where many downstream differences are zero.
    c = (
        _values(rng, (ny, nx))
        if rng.uniform() < 0.5
        else rng.choice(np.array([-2.5, 0.0, -0.0, 1.0, 1.0e3]), size=(ny, nx))
    )
    flux = _planted_flux(shape, rng)
    inflow = _values(rng, shape)
    mask = None if vents == "none" else rng.random(shape) < 0.2
    axis_x, axis_y = grid.axes()
    with np.errstate(over="ignore"):
        if d == 0:
            reference = advective_flux(c, flux, inflow, axis_x, upwind, mask)
        else:
            v = None if mask is None else np.ascontiguousarray(mask.T)
            reference = advective_flux(c.T, flux.T, inflow.T, axis_y, upwind, v).T
    out = np.full(shape, np.nan)
    scheme = _cfdcore.ScalarScheme2(grid.core())
    _cfdcore.advective_flux(scheme, d, c, flux, inflow, upwind, out, mask)
    differ = np.flatnonzero(
        np.ascontiguousarray(reference).view(np.uint64) != out.view(np.uint64)
    )
    assert differ.size == 0, (
        differ[:10],
        reference.flat[differ[:3]],
        out.flat[differ[:3]],
    )
