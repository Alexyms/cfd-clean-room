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

import copy
import dataclasses
import functools
import os
from collections.abc import Callable
from typing import NamedTuple, Self

import numpy as np
import pytest
import yaml

from src.boundary_registry import BoundaryRegistry
from src.boundary_vents import vent_face_masks, vent_runs
from src.config import SimConfig
from src.mesh import SOLID, Mesh
from src.momentum import _lagrange_weights, quick_face_values
from src.scalar_scheme import advective_flux, limited_face_values, mesh_axes

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
