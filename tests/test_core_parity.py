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
import functools
import os

import numpy as np
import pytest
import yaml

from src.boundary_registry import BoundaryRegistry
from src.boundary_vents import vent_face_masks, vent_runs
from src.config import SimConfig
from src.mesh import SOLID, Mesh
from src.scalar_scheme import mesh_axes

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


def _grid_arrays() -> dict[str, object]:
    """A small consistent set of Grid2's arguments, [ny, nx] = [3, 4]."""
    return {
        "x": np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
        "xc": np.array([0.5, 1.5, 2.5, 3.5]),
        "y": np.array([0.0, 1.0, 2.0, 3.0]),
        "yc": np.array([0.5, 1.5, 2.5]),
        "cell_type": np.zeros((3, 4), dtype=np.int32),
        "solid": SOLID,
    }


def test_grid_accepts_consistent_arrays() -> None:
    """The arrays _grid_arrays builds make a grid, so each refusal below is its change's."""
    grid = _cfdcore.Grid2(**_grid_arrays())
    assert grid.shape == (3, 4)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        pytest.param("x", np.arange(6.0), id="faces-too-long"),
        pytest.param("yc", np.array([0.5, 1.5]), id="centres-not-cell-type"),
        pytest.param("xc", np.zeros(0), id="no-cell"),
        pytest.param(
            "cell_type", np.zeros((3, 4), dtype=np.int64), id="cell-type-int64"
        ),
        pytest.param("cell_type", np.zeros((3, 4)), id="cell-type-float"),
        pytest.param(
            "cell_type", np.zeros((4, 3), dtype=np.int32), id="cell-type-transposed"
        ),
        pytest.param("cell_type", np.zeros(12, dtype=np.int32), id="cell-type-flat"),
        pytest.param("x", np.arange(10.0)[::2], id="faces-non-contiguous"),
        pytest.param("x", np.arange(5, dtype=np.float32), id="faces-float32"),
    ],
)
def test_grid_refuses_wrong_arrays(key: str, value: np.ndarray) -> None:
    """A wrong shape, dtype or layout is a TypeError at the binding."""
    arrays = _grid_arrays()
    arrays[key] = value
    with pytest.raises(TypeError):
        _cfdcore.Grid2(**arrays)


def test_grid_refuses_a_direction_it_does_not_have() -> None:
    """A direction other than 0 or 1 is a ValueError."""
    grid = _cfdcore.Grid2(**_grid_arrays())
    with pytest.raises(ValueError):
        grid.nodes(2)
