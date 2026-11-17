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


# ---- The scalar scheme's explicit flux (prompt 87) ---------------------------

# The planted fields: random values of both signs over twelve decades; a
# uniform field (exact under every rule); a field of a few levels, so many
# downstream differences are zero; a smooth field with steps and spikes beside
# and inside the SOLID cells; and small values with signed zeros, where the
# sign of a zero psi reaches the face value.
FIELDS = ["random", "uniform", "levels", "steps-spikes", "signed-zeros"]


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


@pytest.mark.parametrize("shape", [(1,), (7,), (1000,), (30, 79), (119, 320)])
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


def _flux_arguments(d: int = 0) -> dict[str, object]:
    """Consistent advective_flux arguments on the 40x15 open mesh."""
    mesh = _mesh("40x15-uniform-open")
    shape = _face_shape(mesh, d)
    return {
        "scheme": _scheme("40x15-uniform-open"),
        "direction": d,
        "c": np.ones(mesh.cell_type.shape),
        "flux": np.ones(shape),
        "inflow": np.ones(shape),
        "upwind": False,
        "out": np.zeros(shape),
        "vents": np.zeros(shape, dtype=bool),
    }


def test_advective_flux_accepts_consistent_arguments() -> None:
    """The arguments _flux_arguments builds are accepted, so each refusal is its change's."""
    for d in (0, 1):
        arguments = _flux_arguments(d)
        _cfdcore.advective_flux(**arguments)
        assert (arguments["out"] == 1.0).all()
        _cfdcore.advective_flux_overhead(**arguments)


def _unaligned_like(a: np.ndarray) -> np.ndarray:
    """A copy of a float64 array whose data pointer is one byte past a multiple of eight."""
    raw = np.zeros(a.nbytes + 16, dtype=np.uint8)
    start = (1 - raw.ctypes.data) % 8
    out = raw[start : start + a.nbytes].view(np.float64).reshape(a.shape)
    out[...] = a
    return out


@pytest.mark.parametrize(
    ("key", "change"),
    [
        pytest.param("c", lambda a: a.T.copy(), id="c-transposed"),
        pytest.param("c", lambda a: a.T, id="c-non-contiguous"),
        pytest.param("c", lambda a: a.astype(np.float32), id="c-float32"),
        pytest.param("c", lambda a: a.reshape(-1), id="c-flat"),
        pytest.param(
            "flux",
            lambda a: np.ones((a.shape[0] + 1, a.shape[1] - 1)),
            id="flux-other-faces",
        ),
        pytest.param("flux", lambda a: a.astype(np.int64), id="flux-int64"),
        pytest.param("inflow", lambda a: a[:, :-1], id="inflow-short"),
        pytest.param("inflow", _unaligned_like, id="inflow-unaligned"),
        pytest.param("out", lambda a: a[:-1], id="out-short"),
        pytest.param("vents", lambda a: a.astype(np.uint8), id="vents-uint8"),
        pytest.param("vents", lambda a: a[:, :-1], id="vents-short"),
    ],
)
def test_advective_flux_refuses_wrong_arrays(key: str, change: object) -> None:
    """A wrong shape, dtype, layout or alignment is a TypeError at the binding."""
    arguments = _flux_arguments()
    arguments[key] = change(arguments[key])
    with pytest.raises(TypeError):
        _cfdcore.advective_flux(**arguments)


def test_advective_flux_refuses_an_out_that_is_an_input() -> None:
    """out sharing memory with flux, or a read-only out, is a TypeError."""
    arguments = _flux_arguments()
    arguments["out"] = arguments["flux"]
    with pytest.raises(TypeError, match="shares memory"):
        _cfdcore.advective_flux(**arguments)
    arguments = _flux_arguments()
    arguments["out"].flags.writeable = False
    with pytest.raises(TypeError):
        _cfdcore.advective_flux(**arguments)


def test_advective_flux_refuses_a_direction_it_does_not_have() -> None:
    """A direction other than 0 or 1 is a ValueError."""
    arguments = _flux_arguments()
    arguments["direction"] = 2
    with pytest.raises(ValueError):
        _cfdcore.advective_flux(**arguments)


def test_quick_and_limited_refuse_wrong_arrays() -> None:
    """quick_face_values and limited_face_values refuse wrong shapes and dtypes."""
    mesh = _mesh("40x15-uniform-open")
    c = np.ones(mesh.cell_type.shape)
    lines = np.ones(c.shape[0])
    positive = np.ones(_face_shape(mesh, 0), dtype=bool)
    out = np.zeros(positive.shape)
    scheme = _scheme("40x15-uniform-open")
    _cfdcore.quick_face_values(scheme, 0, c, lines, lines, positive, out)
    with pytest.raises(TypeError):
        _cfdcore.quick_face_values(scheme, 0, c, lines[:-1], lines, positive, out)
    with pytest.raises(TypeError):
        _cfdcore.quick_face_values(scheme, 1, c, lines, lines, positive, out)
    with pytest.raises(TypeError):
        _cfdcore.quick_face_values(
            scheme, 0, c, lines, lines, positive.astype(np.uint8), out
        )
    a = np.ones(10)
    _cfdcore.limited_face_values(a, a, a, a, np.zeros(10))
    with pytest.raises(TypeError):
        _cfdcore.limited_face_values(a, a, a, a[:-1], np.zeros(10))
    with pytest.raises(TypeError):
        _cfdcore.limited_face_values(a, a, a, a.reshape(2, 5), np.zeros(10))
    with pytest.raises(TypeError, match="shares memory"):
        _cfdcore.limited_face_values(a, a, a, a, a)
