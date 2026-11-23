"""The compiled core's momentum predictor against the NumPy reference (REQ-N03 level 1, ADR-017 D).

``_cfdcore.MomentumPredictor2`` against ``MomentumPredictor`` on the steady path,
given the same inputs, every output compared bit for bit: planted flows of both
signs on prompt 87's meshes and the vents', the scalar path, a viscosity field, a
wall viscosity and both, one sweep and several; small grids of every row length
with planted SOLID cells, vents and edges; and the quantities the core fixes at
construction against the values the reference forms inside a call. Then the
refusals, the reference's own and the binding's. Skipped with its reason when
``_cfdcore`` cannot be imported, an error instead with ``CFD_CORE_REQUIRED=1``,
as ``tests/test_core_parity.py``.
"""

import copy
import dataclasses
import functools
import os
import sys
from collections.abc import Callable
from types import CodeType, FrameType, SimpleNamespace

import numpy as np
import pytest

from src import momentum
from src.boundary_staggered import StaggeredBoundary, TangentialCondition
from src.boundary_vents import VentFaces
from src.config import SimConfig
from src.mesh import FLUID, SOLID, Mesh
from src.momentum import MomentumPredictor
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

# The parity meshes, after the skip above: that module needs the core too.
from tests.test_core_parity import (
    ALL_MESHES,
    MESHES,
    VENT_MESHES,
    _config,
    _fortran,
    _mesh,
    _raw_config,
    _strided,
    _unaligned_like,
)

pytestmark = [pytest.mark.parity, pytest.mark.integration]

SEED = 92
# The outputs of a predict call, in the binding's order.
OUTPUTS = ("u_star", "v_star", "a_p_u", "a_p_v")
# What each call is handed besides the fields: air's viscosity, a field, a wall
# viscosity, or both.
VISCOSITIES = ["air", "field", "wall", "field-wall"]
SWEEPS = [1, 2, 3, 10]
FLOWS = ["decades", "wide", "smooth", "zeros"]


# ---- The two predictors ------------------------------------------------------


@dataclasses.dataclass(frozen=True, eq=False)
class Room:
    """A mesh with a velocity boundary, and the two predictors built on it."""

    mesh: object
    reference: MomentumPredictor
    core: "_cfdcore.MomentumPredictor2"


def _core_predictor(
    mesh: object, config: object, boundary: object
) -> "_cfdcore.MomentumPredictor2":
    """The core's predictor on a mesh, as the reference's arguments make it."""
    return core_mapping.momentum_predictor(mesh, config, boundary)


def _room(mesh: object, config: object, boundary: object) -> Room:
    return Room(
        mesh,
        MomentumPredictor(mesh, config, boundary),
        _core_predictor(mesh, config, boundary),
    )


def _solver_config(name: str, sweeps: int, alpha: float | None = None) -> SimConfig:
    """One of ALL_MESHES' configurations with its momentum sweeps and relaxation set."""
    raw = _raw_config(name)
    raw["solver"]["momentum_sweeps"] = sweeps
    if alpha is not None:
        raw["solver"]["alpha_velocity"] = alpha
    return SimConfig.from_dict(raw, "configs")


@functools.cache
def _boundary(name: str) -> StaggeredBoundary:
    return StaggeredBoundary(_mesh(name), _config(name))


class PlantedBoundary:
    """A velocity boundary whose edges and vents are planted.

    Parameters
    ----------
    mesh : Mesh or SimpleNamespace
        Supplies the shapes and the wall distances.
    rng : np.random.Generator
        Draws the edges' conditions and, with ``vents``, the vent faces.
    vents : bool
        Whether a share of the faces bounding one SOLID cell are vents.
    """

    def __init__(
        self, mesh: object, rng: np.random.Generator, vents: bool = False
    ) -> None:
        ny, nx = mesh.cell_type.shape
        self._tangential = {}
        for edge, component, n, distance in (
            ("bottom", "u", nx + 1, mesh.dy_face[0]),
            ("top", "u", nx + 1, mesh.dy_face[ny]),
            ("left", "v", ny + 1, mesh.dx_face[0]),
            ("right", "v", ny + 1, mesh.dx_face[nx]),
        ):
            known = rng.random(n) < 0.6
            value = np.where(known, rng.standard_normal(n), 0.0)
            # A wall at rest among them, of either zero.
            value[rng.random(n) < 0.3] = 0.0
            value[rng.random(n) < 0.1] = -0.0
            self._tangential[edge] = TangentialCondition(
                edge, component, known, value, float(distance)
            )
        solid = mesh.cell_type == SOLID
        normal_u = np.zeros((ny, nx + 1), dtype=bool)
        normal_v = np.zeros((ny + 1, nx), dtype=bool)
        if vents:
            normal_u[:, 1:-1] = (solid[:, 1:] ^ solid[:, :-1]) & (
                rng.random((ny, nx - 1)) < 0.4
            )
            normal_v[1:-1, :] = (solid[1:, :] ^ solid[:-1, :]) & (
                rng.random((ny - 1, nx)) < 0.4
            )
        none = np.zeros((ny + 1, nx + 1), dtype=bool)
        self._vents = VentFaces(normal_u, normal_v, none, none)

    def tangential_conditions(self) -> dict[str, TangentialCondition]:
        """The planted conditions, by edge."""
        return self._tangential

    def vent_faces(self) -> VentFaces:
        """The planted vents."""
        return self._vents


# ---- The planted inputs ------------------------------------------------------


def _decades(
    rng: np.random.Generator, shape: tuple[int, int], decades: float
) -> np.ndarray:
    """Random values of both signs over a range of decades about 1."""
    return rng.standard_normal(shape) * 10.0 ** rng.uniform(-decades, decades, shape)


def _planted_velocity(
    kind: str, shape: tuple[int, int], rng: np.random.Generator
) -> np.ndarray:
    """A planted staggered component of one of FLOWS, both signs in each."""
    if kind == "decades":
        return _decades(rng, shape, 3.0)
    if kind == "wide":
        # Wide enough that where every mass flux enters a control volume its
        # diagonal, the conductances' sum, is lost to the fluxes' rounding: the
        # reference then takes the face for one that is not an unknown.
        return _decades(rng, shape, 14.0)
    if kind == "smooth":
        x, y = np.meshgrid(
            np.linspace(0.0, 1.0, shape[1]), np.linspace(0.0, 1.0, shape[0])
        )
        return 0.45 * np.sin(7.0 * x + 3.0 * y + rng.random())
    if kind == "zeros":
        return rng.choice(np.array([0.0, -0.0, 0.3, -0.2, 1.0e-300]), size=shape)
    raise ValueError(kind)


@dataclasses.dataclass
class Inputs:
    """The arguments of one predict call."""

    u: np.ndarray
    v: np.ndarray
    p: np.ndarray
    mu_eff: np.ndarray | None = None
    wall_mu: dict[str, np.ndarray] | None = None


def _planted(
    room: Room,
    flow: str,
    viscosity: str,
    rng: np.random.Generator,
    solid_faces: str = "zero",
) -> Inputs:
    """Planted inputs of one call.

    ``solid_faces`` "zero" holds the faces of SOLID cells at zero but for the
    vents', as a run's fields do; "any" leaves the planted values there. What
    the reference says it does not read holds a NaN: a SOLID cell's viscosity
    and the wall viscosity of a face no unknown reads.
    """
    solid = room.mesh.cell_type == SOLID
    ny, nx = solid.shape
    u = _planted_velocity(flow, (ny, nx + 1), rng)
    v = _planted_velocity(flow, (ny + 1, nx), rng)
    if solid_faces == "zero":
        held = room.reference._held
        blocked = np.zeros((ny, nx + 1), dtype=bool)
        blocked[:, 1:-1] = solid[:, 1:] | solid[:, :-1]
        u[blocked & ~_held(held["u"], blocked.shape)] = 0.0
        blocked = np.zeros((ny + 1, nx), dtype=bool)
        blocked[1:-1, :] = solid[1:, :] | solid[:-1, :]
        v[blocked & ~_held(held["v"], blocked.T.shape).T] = 0.0
    inputs = Inputs(u, v, _decades(rng, (ny, nx), 2.0))
    if "field" in viscosity:
        field = np.abs(_decades(rng, (ny, nx), 2.0)) * 1.0e-4 + 1.0e-6
        inputs.mu_eff = np.where(solid, np.nan, field)
    if "wall" in viscosity:
        inputs.wall_mu = {}
        for key in ("u", "v"):
            wall = np.abs(_decades(rng, (ny + 1, nx + 1), 2.0)) * 1.0e-4 + 1.0e-6
            inputs.wall_mu[key] = np.where(room.reference._wall_read[key], wall, np.nan)
    return inputs


def _held(mask: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray:
    return np.zeros(shape, dtype=bool) if mask is None else mask


def _reference(room: Room, inputs: Inputs) -> dict[str, np.ndarray]:
    """The reference's outputs by name; NumPy's warnings on planted zeros and NaNs are noise."""
    with np.errstate(all="ignore"):
        out = room.reference.predict(
            inputs.u, inputs.v, inputs.p, inputs.mu_eff, inputs.wall_mu
        )
    return {name: getattr(out, name) for name in OUTPUTS}


def _core(room: Room, inputs: Inputs) -> dict[str, np.ndarray]:
    """The core's outputs by name."""
    out = room.core.predict(inputs.u, inputs.v, inputs.p, inputs.mu_eff, inputs.wall_mu)
    return dict(zip(OUTPUTS, out, strict=True))


def _differ(a: np.ndarray, b: np.ndarray) -> int:
    """How many entries of two float64 arrays differ in their 64 bits; -1 for shapes."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return -1
    a, b = np.ascontiguousarray(a), np.ascontiguousarray(b)
    return int(np.count_nonzero(a.view(np.uint64) != b.view(np.uint64)))


def _assert_bit_for_bit(room: Room, inputs: Inputs) -> dict[str, np.ndarray]:
    """Every output of the core's call is the reference's; returns the reference's."""
    expected = _reference(room, inputs)
    got = _core(room, inputs)
    differ = {name: _differ(expected[name], got[name]) for name in OUTPUTS}
    assert differ == dict.fromkeys(OUTPUTS, 0)
    return expected


# ---- predict -----------------------------------------------------------------


@pytest.mark.parametrize("flow", FLOWS)
@pytest.mark.parametrize("viscosity", VISCOSITIES)
@pytest.mark.parametrize("sweeps", SWEEPS)
@pytest.mark.parametrize("name", list(ALL_MESHES))
def test_predict_equals_the_reference_bit_for_bit(
    name: str, sweeps: int, viscosity: str, flow: str
) -> None:
    """u*, v* and both diagonals under the room's own boundary (prediction (a))."""
    rng = np.random.default_rng(
        [SEED, sweeps, VISCOSITIES.index(viscosity), FLOWS.index(flow), len(name)]
    )
    room = _room(_mesh(name), _solver_config(name, sweeps), _boundary(name))
    _assert_bit_for_bit(room, _planted(room, flow, viscosity, rng))


@pytest.mark.parametrize("viscosity", VISCOSITIES)
@pytest.mark.parametrize("alpha", [0.5, 0.7, 1.0])
@pytest.mark.parametrize("vents", [False, True])
@pytest.mark.parametrize("name", ["80x30", "160x60-stretched", "40x15-uniform-open"])
def test_predict_with_planted_edges_equals_the_reference_bit_for_bit(
    name: str, vents: bool, alpha: float, viscosity: str
) -> None:
    """Edges part Dirichlet and part zero-gradient, planted vents, any value on SOLID faces."""
    rng = np.random.default_rng(
        [SEED, int(vents), int(10 * alpha), VISCOSITIES.index(viscosity), len(name)]
    )
    mesh = _mesh(name)
    room = _room(
        mesh, _solver_config(name, 4, alpha), PlantedBoundary(mesh, rng, vents)
    )
    for solid_faces in ("zero", "any"):
        _assert_bit_for_bit(
            room, _planted(room, "decades", viscosity, rng, solid_faces)
        )


@pytest.mark.parametrize("sweeps", [1, 10])
@pytest.mark.parametrize("name", list(VENT_MESHES))
def test_the_sweeps_keep_the_vents_values(name: str, sweeps: int) -> None:
    """A vent's face leaves a call with the value it had; another SOLID face with zero."""
    rng = np.random.default_rng([SEED, sweeps, len(name)])
    room = _room(_mesh(name), _solver_config(name, sweeps), _boundary(name))
    inputs = _planted(room, "smooth", "field-wall", rng)
    _boundary(name).apply_normal_velocity(inputs.u, inputs.v)
    vents = _boundary(name).vent_faces()
    got = _core(room, inputs)
    _assert_bit_for_bit(room, inputs)
    assert np.any(inputs.u[vents.normal_u] != 0.0)
    assert np.any(inputs.v[vents.normal_v] != 0.0)
    assert np.array_equal(got["u_star"][vents.normal_u], inputs.u[vents.normal_u])
    assert np.array_equal(got["v_star"][vents.normal_v], inputs.v[vents.normal_v])


def _small_room(
    ny: int, nx: int, rng: np.random.Generator, sweeps: int, stretch: bool = True
) -> Room:
    """A small grid of random widths, SOLID cells, edges and vents, and a random relaxation."""

    def axis(n: int) -> tuple[np.ndarray, ...]:
        widths = 0.5 + rng.random(n) if stretch else np.full(n, 0.7)
        faces = np.concatenate(([0.0], np.cumsum(widths)))
        centres = 0.5 * (faces[:-1] + faces[1:])
        nodes = np.concatenate((faces[:1], centres, faces[-1:]))
        return faces, centres, np.diff(faces), np.diff(nodes)

    x, xc, dx_cell, dx_face = axis(nx)
    y, yc, dy_cell, dy_face = axis(ny)
    cell_type = np.full((ny, nx), FLUID, dtype=np.int32)
    cell_type[rng.random((ny, nx)) < 0.2] = SOLID
    mesh = SimpleNamespace(
        x=x,
        xc=xc,
        y=y,
        yc=yc,
        dx_cell=dx_cell,
        dy_cell=dy_cell,
        dx_face=dx_face,
        dy_face=dy_face,
        cell_type=cell_type,
    )
    config = SimpleNamespace(
        rho=1.2,
        mu=1.8e-2,
        alpha_velocity=0.3 + 0.7 * rng.random(),
        momentum_sweeps=sweeps,
    )
    return _room(mesh, config, PlantedBoundary(mesh, rng, vents=True))


ROW_WIDTHS = [*range(2, 20), 26, 33]
ROW_HEIGHTS = [2, 3, 4, 5, 9]


@pytest.mark.parametrize("ny", ROW_HEIGHTS)
@pytest.mark.parametrize("nx", ROW_WIDTHS)
def test_predict_is_bit_for_bit_at_every_row_length(nx: int, ny: int) -> None:
    """Every number of cells left after the blocks, with obstacles, edges and vents anywhere."""
    rng = np.random.default_rng([SEED, nx, ny])
    for viscosity in VISCOSITIES:
        for sweeps in (1, 2, 5):
            room = _small_room(ny, nx, rng, sweeps)
            for solid_faces in ("zero", "any"):
                flow = FLOWS[int(rng.integers(len(FLOWS)))]
                _assert_bit_for_bit(
                    room, _planted(room, flow, viscosity, rng, solid_faces)
                )


def test_one_predictor_through_many_calls_is_each_calls_own() -> None:
    """A call leaves nothing the next one reads: paths and fields alternate on one predictor."""
    name = "80x30-vents"
    rng = np.random.default_rng([SEED, 7])
    room = _room(_mesh(name), _solver_config(name, 3), _boundary(name))
    for viscosity in (*VISCOSITIES, *reversed(VISCOSITIES), "air"):
        _assert_bit_for_bit(room, _planted(room, "decades", viscosity, rng))


def test_the_planted_inputs_reach_every_rule() -> None:
    """The planted cases exercise each choice the predictor makes, so the comparisons can fail.

    Counted on the 80x30 room (its tools stand on the floor) and on the vents'
    with the shelf, whose underside is an obstacle's low side.
    """
    reached: dict[str, int] = {}
    for name, flow in (("80x30", "wide"), ("80x30-vents-shelf", "decades")):
        rng = np.random.default_rng([SEED, len(name)])
        room = _room(_mesh(name), _solver_config(name, 1), _boundary(name))
        inputs = _planted(room, flow, "field-wall", rng)
        traced = _trace(room, inputs)
        for c, (o, deferred) in enumerate(
            zip(
                (room.reference._for_u, room.reference._for_v),
                traced["_deferred_correction"],
                strict=True,
            )
        ):
            pos_s, pos_t = deferred["pos_s"], deferred["pos_t"]
            blocked = traced["_obstacle_streamwise"][c]
            inner = ~o.solid[:, 1:-1]
            counts = {
                "streamwise-pos": int(
                    (pos_s & blocked["blocked_pos"])[:, 1:-1][inner].sum()
                ),
                "streamwise-neg": int(
                    (~pos_s & blocked["blocked_neg"])[:, 1:-1][inner].sum()
                ),
                "wall-pos": int(
                    (pos_t & traced["_obstacle_transverse"][c]["south"][:-1]).sum()
                ),
                "wall-neg": int(
                    (~pos_t & traced["_obstacle_transverse"][c]["north"][1:]).sum()
                ),
                "obstacle-face": int(o.walls.face.sum()),
                "north": int(o.walls.north.sum()),
                "south": int(o.walls.south.sum()),
            }
            for key, count in counts.items():
                reached[f"{key}-{'uv'[c]}"] = reached.get(f"{key}-{'uv'[c]}", 0) + count
        expected = _reference(room, inputs)
        solid = room.mesh.cell_type == SOLID
        open_faces = ~(solid[:, 1:] | solid[:, :-1])
        reached["not-positive"] = reached.get("not-positive", 0) + int(
            (expected["a_p_u"][:, 1:-1][open_faces] <= 0.0).sum()
        )
    assert min(reached.values()) > 0, reached


def test_a_nan_lands_where_the_references_does() -> None:
    """A NaN or an infinity in a field gives NaNs at the reference's faces and no other change."""
    rng = np.random.default_rng([SEED, 11])
    name = "80x30"
    for sweeps in (1, 4):
        room = _room(_mesh(name), _solver_config(name, sweeps), _boundary(name))
        for field in ("u", "v", "p"):
            for value in (np.nan, np.inf):
                inputs = _planted(room, "smooth", "field-wall", rng)
                array = getattr(inputs, field)
                for _ in range(5):
                    array[tuple(int(rng.integers(n)) for n in array.shape)] = value
                expected, got = _reference(room, inputs), _core(room, inputs)
                for out in OUTPUTS:
                    nan = np.isnan(expected[out])
                    assert np.array_equal(nan, np.isnan(got[out])), (field, value, out)
                    assert _differ(expected[out][~nan], got[out][~nan]) == 0
                assert not all(np.isfinite(expected[out]).all() for out in OUTPUTS)


# ---- What is fixed at construction -------------------------------------------


TRACED: dict[CodeType, str] = {
    function.__code__: name
    for name, function in {
        "_assemble": MomentumPredictor._assemble,
        "_deferred_correction": MomentumPredictor._deferred_correction,
        "_obstacle_streamwise": MomentumPredictor._obstacle_streamwise,
        "_obstacle_transverse": MomentumPredictor._obstacle_transverse,
        "_corner_viscosity": MomentumPredictor._corner_viscosity,
        "_face_viscosity": MomentumPredictor._face_viscosity,
        "_lagrange_weights": momentum._lagrange_weights,
        "_harmonic": momentum._harmonic,
        "_lerp": momentum._lerp,
    }.items()
}


def _trace(room: Room, inputs: Inputs) -> dict[str, list[dict[str, object]]]:
    """The reference's own local values in one predict call, by function, in call order.

    Read as each function returns, so a value is the one the call formed; the
    caller's name is kept under "caller" and what it returned under "returned".
    """
    seen: dict[str, list[dict[str, object]]] = {name: [] for name in TRACED.values()}

    def profile(frame: FrameType, event: str, arg: object) -> None:
        name = TRACED.get(frame.f_code)
        if event != "return" or name is None:
            return
        local = dict(frame.f_locals)
        local["caller"] = frame.f_back.f_code.co_name if frame.f_back else ""
        local["returned"] = arg
        seen[name].append(local)

    sys.setprofile(profile)
    try:
        _reference(room, inputs)
    finally:
        sys.setprofile(None)
    return seen


def _in_layout(array: np.ndarray, component: int) -> np.ndarray:
    """An array of the reference's frame in the core's layout."""
    return np.ascontiguousarray(array if component == 0 else array.T)


def _fixed_differences(
    room: Room, trace: dict[str, list[dict[str, object]]], scalar: dict[str, object]
) -> list[str]:
    """The fixed quantities of both components that are not the reference's, by name.

    ``trace`` is a call's with a viscosity field and ``scalar`` one's without,
    whose conductances are the scalar path's.
    """
    bad = []
    weights = {
        caller: [t for t in trace["_lagrange_weights"] if t["caller"] == caller]
        for caller in (
            "quick_face_values",
            "_obstacle_streamwise",
            "_obstacle_transverse",
        )
    }
    obstacles = bool((room.mesh.cell_type == SOLID).any())
    for c in (0, 1):
        o = (room.reference._for_u, room.reference._for_v)[c]
        nt, ns = o.solid.shape

        def check(
            name: str, expected: np.ndarray, part: slice | None = None, c: int = c
        ) -> None:
            got = room.core.fixed(name, c)
            got = got if part is None else got[part]
            if np.asarray(expected).dtype == bool:
                same = got.dtype == np.uint8 and np.array_equal(
                    got, np.asarray(expected, dtype=np.uint8)
                )
            else:
                same = _differ(np.ascontiguousarray(expected), got) == 0
            if not same:
                bad.append(f"{name} {c}")

        def check_weights(
            name: str, returned: tuple, part: slice | None = None, c: int = c
        ) -> None:
            for got, expected in zip(room.core.fixed(name, c), returned, strict=True):
                got = got if part is None else got[part]
                if _differ(np.ascontiguousarray(expected).reshape(-1), got) != 0:
                    bad.append(f"{name} {c}")

        # quick_face_values is called for s and then for t, each forming the
        # weights of the two directions of flow.
        quick = weights["quick_face_values"][4 * c : 4 * c + 4]
        interior = slice(1, nt)
        check_weights("std_pos", quick[0]["returned"])
        check_weights("std_neg", quick[1]["returned"])
        check_weights("t_pos", quick[2]["returned"], interior)
        check_weights("t_neg", quick[3]["returned"], interior)
        unknown = np.zeros((nt, ns + 1), dtype=bool)
        unknown[:, 1:-1] = ~trace["_assemble"][c]["solid_face"]
        check("unknown", _in_layout(unknown, c))
        couples = np.ones((nt, ns + 1), dtype=bool)
        couples[:, 1:-1] = ~o.walls.north
        check("couples_high", _in_layout(couples, c))
        couples = np.ones((nt, ns + 1), dtype=bool)
        couples[:, 1:-1] = ~o.walls.south
        check("couples_low", _in_layout(couples, c))
        if obstacles:
            stream = weights["_obstacle_streamwise"][2 * c : 2 * c + 2]
            check_weights("alt_pos", stream[0]["returned"])
            check_weights("alt_neg", stream[1]["returned"])
            cross = weights["_obstacle_transverse"][2 * c : 2 * c + 2]
            check_weights("wall_pos", cross[0]["returned"][1:], interior)
            check_weights("wall_neg", cross[1]["returned"][1:], interior)
            local = trace["_obstacle_streamwise"][c]
            check("blocked_pos", _in_layout(local["blocked_pos"], c))
            check("blocked_neg", _in_layout(local["blocked_neg"], c))
            local = trace["_obstacle_transverse"][c]
            far = np.zeros((nt + 1, ns + 1), dtype=bool)
            far[1:-1] = local["south"][:-1]
            check("far_wall_low", _in_layout(far, c))
            far = np.zeros((nt + 1, ns + 1), dtype=bool)
            far[1:-1] = local["north"][1:]
            check("far_wall_high", _in_layout(far, c))
        else:
            for name in ("blocked_pos", "blocked_neg", "far_wall_low", "far_wall_high"):
                if room.core.fixed(name, c).any():
                    bad.append(f"{name} {c}")
        check("wall_face", _in_layout(o.walls.face, c))
        key = "uv"[c]
        check("wall_read", room.reference._wall_read[key])
        edge = np.zeros((nt + 1, ns + 1), dtype=bool)
        edge[0], edge[-1] = o.low.is_dirichlet, o.high.is_dirichlet
        check("takes_wall", _in_layout(edge | o.walls.face, c))
        conducts = np.ones((nt + 1, ns + 1), dtype=bool)
        conducts[0], conducts[-1] = o.low.is_dirichlet, o.high.is_dirichlet
        check("conducts", _in_layout(conducts, c))
        reach = np.repeat(o.dt_face[:, None], ns + 1, axis=1)
        reach[0], reach[-1] = o.low.wall_distance, o.high.wall_distance
        check("reach", _in_layout(np.where(o.walls.face, o.walls.distance, reach), c))
        corner = trace["_corner_viscosity"][c]
        harmonic = trace["_harmonic"][c]
        check("harmonic_low", corner["frac_lo"][:, 0], interior)
        check("harmonic_low", harmonic["frac_a"][:, 0], interior)
        check("harmonic_high", harmonic["frac_b"][:, 0], interior)
        # _lerp is called by the corner's and then the face's mean.
        for lerp in trace["_lerp"][2 * c : 2 * c + 2]:
            check("mean_high", lerp["frac_b"][0], slice(1, ns))
        check("conductance_s", _in_layout(scalar["_assemble"][c]["d_s"], c))
        check("conductance_t", _in_layout(scalar["_assemble"][c]["d_t"], c))
    return bad


def _assert_fixed(room: Room, rng: np.random.Generator) -> None:
    trace = _trace(room, _planted(room, "smooth", "field-wall", rng))
    scalar = _trace(room, _planted(room, "smooth", "air", rng))
    assert _fixed_differences(room, trace, scalar) == []


@pytest.mark.parametrize("edges", ["own", "planted"])
@pytest.mark.parametrize("name", list(ALL_MESHES))
def test_the_fixed_quantities_equal_the_references_per_call_values(
    name: str, edges: str
) -> None:
    """Each quantity the core fixes is the one the reference forms in a call (prediction (b))."""
    rng = np.random.default_rng([SEED, 3, len(name)])
    mesh = _mesh(name)
    boundary = _boundary(name) if edges == "own" else PlantedBoundary(mesh, rng)
    _assert_fixed(_room(mesh, _solver_config(name, 1), boundary), rng)


@pytest.mark.parametrize("ny", ROW_HEIGHTS)
@pytest.mark.parametrize("nx", ROW_WIDTHS)
def test_the_fixed_quantities_at_every_row_length(nx: int, ny: int) -> None:
    """The same on small grids, down to two cells along an axis."""
    rng = np.random.default_rng([SEED, 4, nx, ny])
    _assert_fixed(_small_room(ny, nx, rng, 1), rng)


def _reference_coefficients(
    room: Room, component: int, inputs: Inputs
) -> momentum.MomentumCoefficients:
    """One component's equations as predict assembles them, in the core's layout."""
    reference = room.reference
    mu = None if inputs.mu_eff is None else reference._check_mu_eff(inputs.mu_eff)
    walls = inputs.wall_mu
    with np.errstate(all="ignore"):
        if component == 0:
            return reference._assemble(
                inputs.u,
                inputs.v,
                reference._for_u,
                mu,
                None if walls is None else walls["u"],
            )
        return momentum._transpose(
            reference._assemble(
                inputs.v.T,
                inputs.u.T,
                reference._for_v,
                None if mu is None else np.ascontiguousarray(mu.T),
                None if walls is None else np.ascontiguousarray(walls["v"].T),
            )
        )


@pytest.mark.parametrize("viscosity", VISCOSITIES)
@pytest.mark.parametrize("name", list(MESHES))
def test_the_coefficients_equal_the_references(name: str, viscosity: str) -> None:
    """The neighbour coefficients and the diagonal of each component, every path."""
    rng = np.random.default_rng([SEED, 5, VISCOSITIES.index(viscosity), len(name)])
    mesh = _mesh(name)
    room = _room(mesh, _solver_config(name, 1), PlantedBoundary(mesh, rng))
    inputs = _planted(room, "decades", viscosity, rng)
    # Twice round, so each component follows the other in the shared arrays.
    for c in (1, 0, 1):
        expected = _reference_coefficients(room, c, inputs)
        got = room.core.coefficients(
            c, inputs.u, inputs.v, inputs.p, inputs.mu_eff, inputs.wall_mu
        )
        for key in ("a_p", "a_s_plus", "a_s_minus", "a_t_plus", "a_t_minus"):
            assert _differ(getattr(expected, key), got[key]) == 0, (c, key)


# ---- stencil_viscosity -------------------------------------------------------


def _assert_stencil(room: Room, mu_eff: np.ndarray) -> None:
    expected = room.reference.stencil_viscosity(mu_eff)
    got = room.core.stencil_viscosity(mu_eff)
    assert set(got) == {"u", "v"}
    assert {key: _differ(expected[key], got[key]) for key in got} == {"u": 0, "v": 0}


@pytest.mark.parametrize("edges", ["own", "planted"])
@pytest.mark.parametrize("name", list(ALL_MESHES))
def test_stencil_viscosity_equals_the_reference_bit_for_bit(
    name: str, edges: str
) -> None:
    """The wall stencil's own viscosity, per face of both components."""
    rng = np.random.default_rng([SEED, 6, len(name)])
    mesh = _mesh(name)
    boundary = _boundary(name) if edges == "own" else PlantedBoundary(mesh, rng)
    room = _room(mesh, _solver_config(name, 2), boundary)
    mu_eff = _planted(room, "smooth", "field", rng).mu_eff
    _assert_stencil(room, mu_eff)
    # As the solver calls the two: predict given it is predict without wall_mu.
    inputs = _planted(room, "smooth", "field", rng)
    inputs.wall_mu = room.core.stencil_viscosity(inputs.mu_eff)
    with_walls = _assert_bit_for_bit(room, inputs)
    inputs.wall_mu = None
    without = _core(room, inputs)
    assert all(_differ(with_walls[name], without[name]) == 0 for name in OUTPUTS)


@pytest.mark.parametrize("nx", ROW_WIDTHS)
def test_stencil_viscosity_is_bit_for_bit_at_every_row_length(nx: int) -> None:
    """The same on small grids of every row length."""
    rng = np.random.default_rng([SEED, 8, nx])
    for ny in ROW_HEIGHTS:
        room = _small_room(ny, nx, rng, 1)
        _assert_stencil(room, _planted(room, "smooth", "field", rng).mu_eff)


# ---- The refusals ------------------------------------------------------------


@functools.cache
def _refusal_room() -> Room:
    return _room(_mesh("80x30"), _solver_config("80x30", 2), _boundary("80x30"))


def _good() -> Inputs:
    """One call's inputs, valid in every argument."""
    return _planted(
        _refusal_room(), "smooth", "field-wall", np.random.default_rng(SEED)
    )


def _last(mask: np.ndarray) -> tuple[int, ...]:
    """The index of a mask's last set entry, so a check that stops early misses it."""
    return tuple(int(i) for i in np.argwhere(mask)[-1])


def _set(array: np.ndarray, where: np.ndarray, value: float) -> None:
    array[_last(where)] = value


def _wall_where(key: str) -> np.ndarray:
    return _refusal_room().reference._wall_read[key]


def _live() -> np.ndarray:
    return _refusal_room().mesh.cell_type != SOLID


# Each case spoils one argument of a good call in place.
REFERENCE_REFUSALS: dict[str, Callable[[Inputs], None]] = {
    "u-shape": lambda i: setattr(i, "u", i.u[:, :-1].copy()),
    "v-shape": lambda i: setattr(i, "v", i.v[:-1].copy()),
    "u-v-swapped": lambda i: (setattr(i, "u", i.v), setattr(i, "v", i.u)),
    "u-one-dimension": lambda i: setattr(i, "u", i.u[0].copy()),
    "v-three-dimensions": lambda i: setattr(i, "v", i.v[None].copy()),
    "p-shape": lambda i: setattr(i, "p", i.p[:-1].copy()),
    "p-transposed": lambda i: setattr(i, "p", i.p.T.copy()),
    "mu-list": lambda i: setattr(i, "mu_eff", i.mu_eff.tolist()),
    "mu-float": lambda i: setattr(i, "mu_eff", 1.0e-5),
    "mu-float32": lambda i: setattr(i, "mu_eff", i.mu_eff.astype(np.float32)),
    "mu-int": lambda i: setattr(i, "mu_eff", np.ones(i.mu_eff.shape, dtype=np.int64)),
    "mu-big-endian": lambda i: setattr(i, "mu_eff", i.mu_eff.astype(">f8")),
    "mu-shape": lambda i: setattr(i, "mu_eff", i.mu_eff[:, :-1].copy()),
    "mu-transposed": lambda i: setattr(i, "mu_eff", i.mu_eff.T.copy()),
    "mu-one-dimension": lambda i: setattr(i, "mu_eff", i.mu_eff.reshape(-1).copy()),
    "mu-nan": lambda i: _set(i.mu_eff, _live(), np.nan),
    "mu-inf": lambda i: _set(i.mu_eff, _live(), np.inf),
    "mu-minus-inf": lambda i: _set(i.mu_eff, _live(), -np.inf),
    "mu-zero": lambda i: _set(i.mu_eff, _live(), 0.0),
    "mu-negative": lambda i: _set(i.mu_eff, _live(), -1.0e-5),
    "mu-negative-and-nan": lambda i: (
        _set(i.mu_eff, _live(), np.nan),
        i.mu_eff.__setitem__((1, 1), -1.0),
    ),
    "wall-list": lambda i: setattr(i, "wall_mu", [i.wall_mu["u"], i.wall_mu["v"]]),
    "wall-array": lambda i: setattr(i, "wall_mu", i.wall_mu["u"]),
    "wall-missing-key": lambda i: i.wall_mu.pop("v"),
    "wall-extra-key": lambda i: i.wall_mu.update(w=i.wall_mu["u"]),
    "wall-other-key": lambda i: i.wall_mu.update(w=i.wall_mu.pop("v")),
    "wall-empty": lambda i: i.wall_mu.clear(),
    **{
        f"wall-{key}-{what}": functools.partial(change, key)
        for key in ("u", "v")
        for what, change in {
            "list": lambda k, i: i.wall_mu.update({k: i.wall_mu[k].tolist()}),
            "none": lambda k, i: i.wall_mu.update({k: None}),
            "float32": lambda k, i: i.wall_mu.update(
                {k: i.wall_mu[k].astype(np.float32)}
            ),
            "shape": lambda k, i: i.wall_mu.update({k: i.wall_mu[k][:-1].copy()}),
            "cell-shape": lambda k, i: i.wall_mu.update({k: i.mu_eff.copy()}),
            "nan": lambda k, i: _set(i.wall_mu[k], _wall_where(k), np.nan),
            "inf": lambda k, i: _set(i.wall_mu[k], _wall_where(k), np.inf),
            "zero": lambda k, i: _set(i.wall_mu[k], _wall_where(k), 0.0),
            "negative": lambda k, i: _set(i.wall_mu[k], _wall_where(k), -1.0),
        }.items()
    },
    # Two wrong arguments: the reference's first check is the one that answers.
    "u-shape-and-p-shape": lambda i: (
        setattr(i, "u", i.u[:-1].copy()),
        setattr(i, "p", i.p[:-1].copy()),
    ),
    "mu-nan-and-wall-list": lambda i: (
        _set(i.mu_eff, _live(), np.nan),
        setattr(i, "wall_mu", []),
    ),
    "wall-u-nan-and-v-shape": lambda i: (
        _set(i.wall_mu["u"], _wall_where("u"), np.nan),
        i.wall_mu.update(v=i.wall_mu["v"][:-1].copy()),
    ),
}


@pytest.mark.parametrize("case", list(REFERENCE_REFUSALS))
def test_predict_refuses_what_the_reference_refuses(case: str) -> None:
    """The same type and the same message, for every argument the reference refuses."""
    room = _refusal_room()
    inputs = _good()
    REFERENCE_REFUSALS[case](inputs)
    with pytest.raises((TypeError, ValueError)) as expected:
        room.reference.predict(
            inputs.u, inputs.v, inputs.p, inputs.mu_eff, inputs.wall_mu
        )
    with pytest.raises((TypeError, ValueError)) as got:
        room.core.predict(inputs.u, inputs.v, inputs.p, inputs.mu_eff, inputs.wall_mu)
    assert got.type is expected.type
    assert str(got.value) == str(expected.value)


def test_predict_accepts_the_arguments_the_refusals_spoil() -> None:
    """A good call is accepted, so each refusal above is its own change's."""
    _assert_bit_for_bit(_refusal_room(), _good())


@pytest.mark.parametrize(
    "case",
    [n for n in REFERENCE_REFUSALS if n.startswith("mu-") and "wall" not in n],
)
def test_stencil_viscosity_refuses_what_the_reference_refuses(case: str) -> None:
    """The same type and message on mu_eff, None among them."""
    room = _refusal_room()
    inputs = _good()
    REFERENCE_REFUSALS[case](inputs)
    for mu_eff in (inputs.mu_eff, None):
        with pytest.raises((TypeError, ValueError)) as expected:
            room.reference.stencil_viscosity(mu_eff)
        with pytest.raises((TypeError, ValueError)) as got:
            room.core.stencil_viscosity(mu_eff)
        assert got.type is expected.type
        assert str(got.value) == str(expected.value)


@pytest.mark.parametrize("levels", [1, 2])
def test_predict_refuses_a_transient_argument(levels: int) -> None:
    """The transient prediction stays on the reference (ADR-017 F), which accepts this one."""
    room = _refusal_room()
    inputs = _good()
    old = (inputs.u, inputs.v) if levels == 1 else ((inputs.u,) * 2, (inputs.v,) * 2)
    transient = (0.01, *old)
    room.reference.predict(inputs.u, inputs.v, inputs.p, transient=transient)
    with pytest.raises(NotImplementedError, match="steady path"):
        room.core.predict(inputs.u, inputs.v, inputs.p, transient=transient)


BINDING_REFUSALS: dict[str, tuple[Callable[[np.ndarray], np.ndarray], str]] = {
    "float32": (lambda a: a.astype(np.float32), "incompatible function arguments"),
    "fortran": (_fortran, "incompatible function arguments"),
    "strided": (_strided, "incompatible function arguments"),
    "unaligned": (_unaligned_like, "not aligned"),
}


@pytest.mark.parametrize("case", list(BINDING_REFUSALS))
@pytest.mark.parametrize("field", ["u", "v", "p"])
def test_predict_refuses_a_field_it_cannot_read(field: str, case: str) -> None:
    """A wrong dtype, layout or alignment of u, v or p is a TypeError at the call."""
    change, message = BINDING_REFUSALS[case]
    inputs = _good()
    setattr(inputs, field, change(getattr(inputs, field)))
    with pytest.raises(TypeError, match=message):
        _refusal_room().core.predict(inputs.u, inputs.v, inputs.p)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (_fortran, "C-contiguous"),
        (_strided, "C-contiguous"),
        (_unaligned_like, "not aligned"),
    ],
    ids=["fortran", "strided", "unaligned"],
)
@pytest.mark.parametrize("which", ["mu_eff", "u", "v"])
def test_predict_refuses_a_viscosity_it_cannot_read(
    which: str, change: Callable[[np.ndarray], np.ndarray], message: str
) -> None:
    """A viscosity whose layout the core does not read is a TypeError naming it."""
    inputs = _good()
    if which == "mu_eff":
        inputs.mu_eff = change(inputs.mu_eff)
    else:
        inputs.wall_mu[which] = change(inputs.wall_mu[which])
    with pytest.raises(TypeError, match=message):
        _core(_refusal_room(), inputs)


def _arguments() -> dict[str, object]:
    """The constructor's arguments on the 80x30 room, valid in every one."""
    mesh, config, boundary = _mesh("80x30"), _config("80x30"), _boundary("80x30")
    ny, nx = mesh.cell_type.shape
    grid = _cfdcore.Grid2(mesh.x, mesh.xc, mesh.y, mesh.yc, mesh.cell_type, SOLID)
    edges = boundary.tangential_conditions()
    return {
        "grid": grid,
        "dx_cell": mesh.dx_cell,
        "dy_cell": mesh.dy_cell,
        "dx_face": mesh.dx_face,
        "dy_face": mesh.dy_face,
        "rho": config.rho,
        "mu": config.mu,
        "alpha": config.alpha_velocity,
        "sweeps": 1,
        "tangential": [
            (edges[e].is_dirichlet, edges[e].value, edges[e].wall_distance)
            for e in ("bottom", "top", "left", "right")
        ],
        "held_u": np.zeros((ny, nx + 1), dtype=bool),
        "held_v": np.zeros((ny + 1, nx), dtype=bool),
    }


def _spoil_edge(
    arguments: dict[str, object], edge: int, part: int, change: Callable
) -> None:
    """Replace one part of one edge's condition."""
    edges = copy.copy(arguments["tangential"])
    condition = list(edges[edge])
    condition[part] = change(condition[part])
    edges[edge] = tuple(condition)
    arguments["tangential"] = edges


CONSTRUCTOR_REFUSALS: dict[str, tuple[Callable, type[Exception], str]] = {
    **{
        f"{name}-{what}": (
            functools.partial(lambda n, f, a: a.__setitem__(n, f(a[n])), name, change),
            error,
            message,
        )
        for name in ("dx_cell", "dy_cell", "dx_face", "dy_face")
        for what, change, error, message in (
            ("short", lambda a: a[:-1].copy(), ValueError, "are not one per"),
            (
                "long",
                lambda a: np.append(a, 1.0),
                ValueError,
                "are not one per",
            ),
            (
                "float32",
                lambda a: a.astype(np.float32),
                TypeError,
                "incompatible function arguments",
            ),
            (
                "unaligned",
                lambda a: _unaligned_like(a[None])[0],
                TypeError,
                "not aligned",
            ),
        )
    },
    "sweeps-zero": (
        lambda a: a.__setitem__("sweeps", 0),
        ValueError,
        "sweeps is below 1",
    ),
    "three-edges": (
        lambda a: a.__setitem__("tangential", a["tangential"][:3]),
        TypeError,
        "bottom, top, left and right",
    ),
    **{
        f"edge-{edge}-{what}": (
            functools.partial(
                lambda e, p, f, a: _spoil_edge(a, e, p, f), edge, part, change
            ),
            TypeError,
            message,
        )
        for edge, name in enumerate(("bottom", "top", "left", "right"))
        for what, part, change, message in (
            ("flags-short", 0, lambda x: x[:-1].copy(), name),
            ("flags-int", 0, lambda x: x.astype(np.int8), name),
            ("value-short", 1, lambda x: x[:-1].copy(), name),
            ("value-float32", 1, lambda x: x.astype(np.float32), name),
            (
                "value-unaligned",
                1,
                lambda x: _unaligned_like(x[None])[0],
                "not aligned",
            ),
        )
    },
    **{
        f"{name}-{what}": (
            functools.partial(lambda n, f, a: a.__setitem__(n, f(a[n])), name, change),
            TypeError,
            message,
        )
        for name in ("held_u", "held_v")
        for what, change, message in (
            ("transposed", lambda x: x.T.copy(), name),
            ("int", lambda x: x.astype(np.int8), "incompatible function arguments"),
        )
    },
}


@pytest.mark.parametrize("case", list(CONSTRUCTOR_REFUSALS))
def test_the_constructor_refuses_wrong_arguments(case: str) -> None:
    """A wrong length, dtype, shape or count is refused when the predictor is built."""
    change, error, message = CONSTRUCTOR_REFUSALS[case]
    arguments = _arguments()
    change(arguments)
    with pytest.raises(error, match=message):
        _cfdcore.MomentumPredictor2(**arguments)


def test_the_constructor_accepts_the_arguments_the_refusals_spoil() -> None:
    """The arguments of _arguments make a predictor, so each refusal is its own change's."""
    _cfdcore.MomentumPredictor2(**_arguments())


@pytest.mark.parametrize(("ny", "nx"), [(1, 3), (3, 1), (1, 1)])
def test_the_constructor_refuses_fewer_than_two_cells_as_the_reference_does(
    ny: int, nx: int
) -> None:
    """The reference's ValueError and its message."""
    x, y = np.arange(nx + 1.0), np.arange(ny + 1.0)
    xc, yc = 0.5 * (x[1:] + x[:-1]), 0.5 * (y[1:] + y[:-1])
    mesh = SimpleNamespace(
        x=x,
        xc=xc,
        y=y,
        yc=yc,
        dx_cell=np.diff(x),
        dy_cell=np.diff(y),
        dx_face=np.ones(nx + 1),
        dy_face=np.ones(ny + 1),
        cell_type=np.full((ny, nx), FLUID, dtype=np.int32),
    )
    boundary = PlantedBoundary(mesh, np.random.default_rng(SEED))
    config = _config("80x30")
    with pytest.raises(ValueError) as expected:
        MomentumPredictor(mesh, config, boundary)
    with pytest.raises(ValueError) as got:
        _core_predictor(mesh, config, boundary)
    assert str(got.value) == str(expected.value)


def test_fixed_and_coefficients_refuse_names_and_components_they_do_not_have() -> None:
    """An unknown name or component is a ValueError."""
    core = _refusal_room().core
    inputs = _good()
    with pytest.raises(ValueError, match="no quantity is named"):
        core.fixed("weights", 0)
    for component in (-1, 2):
        with pytest.raises(ValueError, match="direction"):
            core.fixed("reach", component)
        with pytest.raises(ValueError, match="direction"):
            core.coefficients(component, inputs.u, inputs.v, inputs.p)


def test_the_binding_needs_nothing_from_src(monkeypatch: pytest.MonkeyPatch) -> None:
    """The predictor is built and called with every src module hidden (ADR-017 B)."""
    arguments = _arguments()
    inputs = _good()
    for name in [n for n in sys.modules if n == "src" or n.startswith("src.")]:
        monkeypatch.setitem(sys.modules, name, None)
    core = _cfdcore.MomentumPredictor2(**arguments)
    out = core.predict(inputs.u, inputs.v, inputs.p, inputs.mu_eff, inputs.wall_mu)
    assert len(out) == 4
    assert set(core.stencil_viscosity(inputs.mu_eff)) == {"u", "v"}


def test_the_outputs_are_new_arrays_and_the_inputs_untouched() -> None:
    """A call writes to none of its inputs, and the next call's outputs are other arrays."""
    room = _refusal_room()
    inputs = _good()
    before = copy.deepcopy(inputs)
    first = _core(room, inputs)
    kept = {name: first[name].copy() for name in OUTPUTS}
    _core(room, _planted(room, "decades", "air", np.random.default_rng(SEED + 1)))
    assert all(_differ(first[name], kept[name]) == 0 for name in OUTPUTS)
    for name in ("u", "v", "p", "mu_eff"):
        assert _differ(getattr(before, name), getattr(inputs, name)) == 0
    for key in ("u", "v"):
        assert _differ(before.wall_mu[key], inputs.wall_mu[key]) == 0


def test_the_meshes_hold_what_the_cases_need() -> None:
    """The stretched mesh is not uniform, the vents' rooms have vents, the last has no obstacle."""
    for name in VENT_MESHES:
        vents = _boundary(name).vent_faces()
        assert vents.normal_u.any() and vents.normal_v.any()
    mesh: Mesh = _mesh("160x60-stretched")
    assert mesh.dx_cell.max() / mesh.dx_cell.min() > 1.5
    assert (mesh.cell_type == SOLID).any()
    assert not (_mesh("40x15-uniform-open").cell_type == SOLID).any()
