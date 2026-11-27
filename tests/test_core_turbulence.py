"""The compiled core's turbulence model and wall functions against the NumPy reference.

REQ-N03 level 1, ADR-017 D; prompt 94. ``_cfdcore.KEpsilon2`` against
``KEpsilonModel`` on the steady path and ``_cfdcore.TurbulenceBoundary2`` against
``TurbulenceBoundary``, block by block given the same inputs: everything
elementwise bit for bit, the set-up's fixed arrays against the values the
reference forms, and the implicit solves by the conjugate gradients' rule (the
reference's count, the field within 1e-12 of its largest magnitude, a fallback
where the reference falls back). Both variants, RNG's R of both signs, upwind and
UMIST advection, on prompt 87's meshes, the Annex 20 mesh, a cavity with a moving
wall and small rooms of planted SOLID cells, edges and vents. Then the refusals,
the reference's own and the binding's.

The elementary functions. NumPy forms ``np.log`` and ``x**3`` with loops of its own
on a processor with AVX-512, which differ from the C library's in the last bit; the
core calls the C library (Alex, 2026-11-25). The tests that need NumPy to call it
too ask for ``libm_numpy`` and run where NumPy's AVX-512 loops are off or absent
(``NPY_DISABLE_CPU_FEATURES=X86_V4``, as CI's core job sets it); elsewhere they skip
with that reason, or fail with ``CFD_CORE_REQUIRED=1``. What those loops change is
tested in every environment at REQ-N03's 1e-12.

Skipped with its reason when ``_cfdcore`` cannot be imported, an error instead with
``CFD_CORE_REQUIRED=1``, as ``tests/test_core_parity.py``.
"""

import copy
import dataclasses
import functools
import math
import os
from collections.abc import Callable
from types import SimpleNamespace

import numpy as np
import pytest
from numpy.lib import introspect

from src import core_backend, turbulence
from src.boundary_staggered import StaggeredBoundary
from src.boundary_vents import VentFaces, vent_face_masks, vent_runs
from src.config import RNG, STANDARD, UMIST, UPWIND, SimConfig
from src.mesh import SOLID, Mesh
from src.scalar_scheme import CONJUGATE_GRADIENT, ImplicitResult, implicit_step
from src.staggered import FaceVelocities
from src.turbulence import (
    E_WALL,
    KAPPA,
    Y_STAR_FLOOR,
    KEpsilonModel,
    PositivityError,
    TurbulenceBoundary,
    TurbulenceConditions,
    TurbulenceState,
    inflow_values,
    log_law_floor,
    wall_viscosity,
)
from validation.cases import _raw_case, load_annex20

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
    _fortran,
    _raw_config,
    _strided,
    _unaligned_like,
)

pytestmark = [pytest.mark.parity, pytest.mark.integration]

SEED = 94
VARIANTS = [STANDARD, RNG]
SCHEMES = [UPWIND, UMIST]
FLOWS = ["decades", "smooth", "zeros", "resting"]
# The flows of a whole step: at the 1e-300 m/s of "zeros" a cell's step overflows.
STEP_FLOWS = ["decades", "smooth", "resting"]
FIELD_TOL = 1e-12
TERMS = ("production", "rng_r", "growth_k", "growth_eps", "decay_k", "decay_eps")
ROOMS = [*ALL_MESHES, "annex20", "cavity"]
CONDITIONS = tuple(f.name for f in dataclasses.fields(TurbulenceConditions))


def _numpy_calls_libm() -> bool:
    """Whether this process's NumPy forms float64 logs and powers with the C library."""
    info = introspect.opt_func_info(func_name="^(log|power)$", signature="float64")
    current = [loop["current"] for loops in info.values() for loop in loops.values()]
    return not any("V4" in name or "AVX512" in name for name in current)


@pytest.fixture
def libm_numpy() -> None:
    """Require a NumPy whose logarithm and powers are the C library's."""
    if _numpy_calls_libm():
        return
    reason = (
        "NumPy forms np.log and x**3 with its own AVX-512 loops here: "
        "set NPY_DISABLE_CPU_FEATURES=X86_V4 (prompt 94)"
    )
    if os.environ.get("CFD_CORE_REQUIRED") == "1":
        pytest.fail(reason)
    pytest.skip(reason)


# ---- The rooms ---------------------------------------------------------------


@dataclasses.dataclass(frozen=True, eq=False)
class Room:
    """A mesh and its configuration, and the two sides' models and boundaries on it."""

    mesh: Mesh
    config: SimConfig
    boundary: StaggeredBoundary
    model: KEpsilonModel
    walls: TurbulenceBoundary
    core: "_cfdcore.KEpsilon2"
    core_walls: "_cfdcore.TurbulenceBoundary2"

    @property
    def solid(self) -> np.ndarray:
        """The SOLID mask."""
        return self.mesh.cell_type == SOLID


TURBULENCE = {
    "model": "k_epsilon",
    "variant": STANDARD,
    "wall_treatment": "scalable_wall_functions",
    "cfl_number": 0.25,
    "advection_scheme": UPWIND,
    "alpha_turbulence": 0.7,
    "max_iter": 500,
    "tol": 1.0e-10,
}


def _raw(name: str) -> dict:
    """One of ROOMS' configurations as a mapping, before validation."""
    if name == "annex20":
        return load_annex20_raw()
    if name == "cavity":
        raw = _raw_case("cavity", (24, 24))
        raw["turbulence"] = dict(TURBULENCE)
        # A wall moving at each edge, of either sign.
        raw["boundaries"] = {
            edge: {
                "type": "velocity_inlet",
                "location": edge,
                **({"x_start": 0.0, "x_end": 1.0} if flat else {}),
                **({} if flat else {"y_start": 0.0, "y_end": 1.0}),
                "u_velocity": speed if flat else 0.0,
                "v_velocity": 0.0 if flat else speed,
            }
            for edge, flat, speed in (
                ("top", True, 1.0),
                ("bottom", True, -0.4),
                ("left", False, 0.3),
                ("right", False, -0.7),
            )
        }
        return raw
    return _raw_config(name)


def load_annex20_raw() -> dict:
    """The Annex 20 case's mapping with the inlet's velocity scaled as ``load_annex20`` does."""
    raw = _raw_case("annex20", (216, 72))
    scaled = load_annex20("annex20", (216, 72))
    raw["boundaries"]["inlet"]["velocity"] = scaled.boundaries["inlet"].velocity
    return raw


def _build(mesh: Mesh, config: SimConfig, boundary: StaggeredBoundary) -> Room:
    grid = core_backend.core_grid(mesh)
    return Room(
        mesh,
        config,
        boundary,
        KEpsilonModel(mesh, config),
        TurbulenceBoundary(mesh, config, boundary),
        core_backend.k_epsilon(mesh, config, boundary, grid),
        core_backend.turbulence_boundary(mesh, config, boundary, grid),
    )


@functools.cache
def _room(name: str, variant: str = STANDARD, scheme: str = UPWIND) -> Room:
    """One of ROOMS with its variant and advection scheme."""
    raw = copy.deepcopy(_raw(name))
    raw["turbulence"].update(variant=variant, advection_scheme=scheme)
    config = SimConfig.from_dict(raw, "configs")
    mesh = Mesh(config)
    return _build(mesh, config, StaggeredBoundary(mesh, config))


# The small rooms' boundary: an inlet on every edge, one of them oblique, a moving
# wall, a pressure outlet on every edge and walls between, on a room 8 m by 3 m. An
# open edge takes the adjacent face's velocity for its tangential velocity, so each
# edge needs one for the rule to be tested on it (review 94, test B1).
def _small_raw(nx: int, ny: int, stretch: float, variant: str, scheme: str) -> dict:
    raw = copy.deepcopy(_raw_config("40x15-uniform-open"))
    raw["domain"].update(nx=nx, ny=ny)
    raw["mesh"]["x"] = {"stretch_ratio": stretch}
    raw["mesh"]["y"] = {"stretch_ratio": stretch}
    raw["turbulence"].update(variant=variant, advection_scheme=scheme)
    inlet = {"turbulence_intensity": 0.07, "dissipation_length": 0.13}
    raw["boundaries"] = {
        "supply": {
            "type": "velocity_inlet",
            "location": "top",
            "x_start": 0.0,
            "x_end": 3.0,
            "velocity": 0.45,
            **inlet,
        },
        "lid": {
            "type": "velocity_inlet",
            "location": "top",
            "x_start": 5.0,
            "x_end": 8.0,
            "u_velocity": -0.6,
            "v_velocity": 0.0,
        },
        "oblique": {
            "type": "velocity_inlet",
            "location": "left",
            "y_start": 0.0,
            "y_end": 1.6,
            "u_velocity": 0.2,
            "v_velocity": 0.1,
            **inlet,
        },
        "below": {
            "type": "velocity_inlet",
            "location": "bottom",
            "x_start": 4.0,
            "x_end": 6.0,
            "u_velocity": 0.05,
            "v_velocity": 0.3,
            "turbulence_intensity": 0.2,
            "dissipation_length": 0.01,
        },
        "exit": {
            "type": "pressure_outlet",
            "location": "right",
            "y_start": 0.8,
            "y_end": 2.2,
        },
        "floor_exit": {
            "type": "pressure_outlet",
            "location": "bottom",
            "x_start": 0.0,
            "x_end": 2.0,
        },
        "roof_exit": {
            "type": "pressure_outlet",
            "location": "top",
            "x_start": 3.2,
            "x_end": 4.8,
        },
        "side_exit": {
            "type": "pressure_outlet",
            "location": "left",
            "y_start": 1.9,
            "y_end": 2.9,
        },
    }
    raw["obstacles"] = []
    return raw


def _small_room(
    size: tuple[int, int],
    variant: str,
    scheme: str,
    rng: np.random.Generator,
    vents: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> Room:
    """A small room with planted SOLID cells and, with ``vents``, planted vents.

    The vents are faces between a SOLID and a room cell, drawn at random, some
    with the stencil's corner over them. The reference's model reads its vents
    through ``vent_runs``, replaced here; its boundary and the core's are handed
    a boundary that states them, with runs that make some discharges and the
    others exhausts.
    """
    nx, ny = size
    config = SimConfig.from_dict(
        _small_raw(nx, ny, 1.0 + 0.1 * rng.random(), variant, scheme), "configs"
    )
    mesh = Mesh(config)
    solid = rng.random((ny, nx)) < 0.25
    solid[rng.integers(ny), rng.integers(nx)] = False
    mesh.cell_type = np.where(solid, SOLID, 0).astype(mesh.cell_type.dtype)
    boundary = StaggeredBoundary(mesh, config)
    if vents:
        boundary = PlantedVents(boundary, solid, rng)
        monkeypatch.setattr(turbulence, "vent_runs", lambda *args: boundary.vent_runs())
        monkeypatch.setattr(
            turbulence, "vent_face_masks", lambda *args: boundary.vent_faces()
        )
    return _build(mesh, config, boundary)


class PlantedVents:
    """A staggered boundary with planted vents (``StaggeredBoundary``'s readers).

    Parameters
    ----------
    boundary : StaggeredBoundary
        Everything but the vents.
    solid : np.ndarray
        The SOLID mask.
    rng : np.random.Generator
        Draws the vents.
    """

    def __init__(
        self, boundary: StaggeredBoundary, solid: np.ndarray, rng: np.random.Generator
    ) -> None:
        self._boundary = boundary
        ny, nx = solid.shape
        u = np.zeros((ny, nx + 1), dtype=bool)
        v = np.zeros((ny + 1, nx), dtype=bool)
        u[:, 1:-1] = (solid[:, 1:] ^ solid[:, :-1]) & (rng.random((ny, nx - 1)) < 0.5)
        v[1:-1, :] = (solid[1:, :] ^ solid[:-1, :]) & (rng.random((ny - 1, nx)) < 0.5)
        # A stencil's corner lies over a vent's span only where there is a vent.
        corners, share = (ny + 1, nx + 1), 0.2 * bool(u.any() or v.any())
        self._faces = VentFaces(
            u, v, rng.random(corners) < share, rng.random(corners) < share
        )
        self._runs = []
        for component, mask in (("u", u), ("v", v)):
            for j, i in np.argwhere(mask):
                inlet = bool(rng.random() < 0.5)
                self._runs.append(
                    SimpleNamespace(
                        name=f"{component}{j}_{i}",
                        component=component,
                        rows=np.array([j]),
                        cols=np.array([i]),
                        widths=np.array([0.1 + rng.random()]),
                        is_inlet=inlet,
                        normal=float(rng.uniform(-0.9, 0.9)) or 0.5,
                        spec=SimpleNamespace(
                            turbulence_intensity=float(rng.uniform(0.01, 0.3)),
                            dissipation_length=float(rng.uniform(0.01, 0.3)),
                        ),
                    )
                )

    def vent_runs(self) -> list[SimpleNamespace]:
        """One run per planted face."""
        return list(self._runs)

    def vent_faces(self) -> VentFaces:
        """The planted masks."""
        return self._faces

    def __getattr__(self, name: str) -> object:
        return getattr(self._boundary, name)


SIZES = [(3, 3), (4, 3), (3, 9), (7, 5), (8, 8), (9, 4), (16, 5), (17, 6), (23, 11)]


@pytest.mark.parametrize("edge", core_backend.WALL_EDGES)
def test_the_small_rooms_are_open_on_every_edge(edge: str) -> None:
    """Most small rooms have a pressure outlet on each edge, so none goes untested.

    An edge's tangential velocity is the adjacent face's where the edge is open and
    the condition's value where it is not. A binding that read one edge's flags as
    all known passed every test while only the bottom and the right edge had an
    outlet (test 94, B1).
    """
    open_rooms = 0
    for nx, ny in SIZES:
        config = SimConfig.from_dict(
            _small_raw(nx, ny, 1.05, STANDARD, UPWIND), "configs"
        )
        edges = StaggeredBoundary(Mesh(config), config).tangential_conditions()
        open_rooms += not edges[edge].is_dirichlet.all()
    assert open_rooms >= 6


# ---- The planted inputs ------------------------------------------------------


def _component(
    kind: str, shape: tuple[int, int], rng: np.random.Generator
) -> np.ndarray:
    """A planted velocity component of one of FLOWS, both signs in each."""
    if kind == "decades":
        return rng.standard_normal(shape) * 10.0 ** rng.uniform(-3.0, 1.0, shape)
    if kind == "smooth":
        x, y = np.meshgrid(np.linspace(0, 1, shape[1]), np.linspace(0, 1, shape[0]))
        return 0.45 * np.sin(7.0 * x + 3.0 * y + rng.random())
    if kind == "zeros":
        return rng.choice(np.array([0.0, -0.0, 0.3, -0.2, 1.0e-300]), size=shape)
    if kind == "resting":
        # Most of the room at rest, so that most cells take the step's cap.
        field = rng.standard_normal(shape) * 0.2
        field[:, shape[1] // 3 :] = 0.0
        return field
    raise ValueError(kind)


@dataclasses.dataclass(frozen=True, eq=False)
class Inputs:
    """The arguments of one step."""

    state: TurbulenceState
    faces: FaceVelocities
    conditions: TurbulenceConditions


def _planted_state(room: Room, rng: np.random.Generator) -> TurbulenceState:
    """k and eps over decades, with an eddy viscosity that is not C_mu k^2 / eps."""
    shape = room.mesh.cell_type.shape
    state = room.model.state(
        10.0 ** rng.uniform(-6.0, 0.0, shape), 10.0 ** rng.uniform(-6.0, 1.0, shape)
    )
    nu_t = np.where(room.solid, 0.0, 10.0 ** rng.uniform(-7.0, -2.0, shape))
    nu_t[rng.random(shape) < 0.05] = 0.0
    return TurbulenceState(k=state.k, eps=state.eps, nu_t=nu_t)


def _planted_conditions(room: Room, rng: np.random.Generator) -> TurbulenceConditions:
    """Conditions with every entry drawn, held and given cells apart from each other."""
    ny, nx = room.mesh.cell_type.shape
    live = ~room.solid
    held = live & (rng.random((ny, nx)) < 0.3)
    given = live & (rng.random((ny, nx)) < 0.3)
    production = 10.0 ** rng.uniform(-5.0, 1.0, (ny, nx))
    production[rng.random((ny, nx)) < 0.1] = 0.0
    edge = rng.standard_normal
    return TurbulenceConditions(
        inflow_k_u=10.0 ** rng.uniform(-5.0, 0.0, (ny, nx + 1)),
        inflow_k_v=10.0 ** rng.uniform(-5.0, 0.0, (ny + 1, nx)),
        inflow_eps_u=10.0 ** rng.uniform(-5.0, 0.0, (ny, nx + 1)),
        inflow_eps_v=10.0 ** rng.uniform(-5.0, 0.0, (ny + 1, nx)),
        tangential_bottom=edge(nx + 1),
        tangential_top=edge(nx + 1),
        tangential_left=edge(ny + 1),
        tangential_right=edge(ny + 1),
        eps_held=held,
        eps_wall=10.0 ** rng.uniform(-5.0, 1.0, (ny, nx)),
        production_given=given,
        production=production,
    )


def _planted(room: Room, flow: str, rng: np.random.Generator) -> Inputs:
    """Planted inputs of one step. The faces of SOLID cells hold values too."""
    ny, nx = room.mesh.cell_type.shape
    u, v = _component(flow, (ny, nx + 1), rng), _component(flow, (ny + 1, nx), rng)
    # At least one live cell moves, so the pseudo-time step has a value.
    j, i = np.argwhere(~room.solid)[0]
    u[j, i : i + 2] = 0.25
    if not room.model._live_u[j, i : i + 2].any():
        v[j : j + 2, i] = -0.25
    return Inputs(
        _planted_state(room, rng),
        FaceVelocities.copy_of(u, v),
        _planted_conditions(room, rng),
    )


def _gentle(room: Room, rng: np.random.Generator) -> Inputs:
    """Inputs a solve of the reference's does not amplify rounding on.

    A smooth flow, k and eps over half a decade, their own eddy viscosity and the
    room's own conditions, as a run's are.
    """
    inputs = _planted(room, "smooth", rng)
    shape = room.mesh.cell_type.shape
    state = room.model.state(
        1.0e-3 * 10.0 ** rng.uniform(-0.25, 0.25, shape),
        1.0e-3 * 10.0 ** rng.uniform(-0.25, 0.25, shape),
    )
    return Inputs(state, inputs.faces, room.walls.conditions(state, inputs.faces))


def _rng(*parts: object) -> np.random.Generator:
    """A generator seeded by the case's own parts."""
    return np.random.default_rng([SEED, *(sum(map(ord, str(p))) for p in parts)])


# ---- Comparisons -------------------------------------------------------------


def _differ(a: object, b: object) -> int:
    """How many entries differ in their bits; -1 for another shape or dtype."""
    a, b = np.ascontiguousarray(a), np.ascontiguousarray(b)
    if a.shape != b.shape or a.dtype != b.dtype:
        return -1
    if a.dtype == np.float64:
        a, b = a.view(np.uint64), b.view(np.uint64)
    return int(np.count_nonzero(a != b))


def _assert_same(expected: dict[str, object], got: dict[str, object]) -> None:
    """Every named array equals the reference's in every bit."""
    assert {name: _differ(value, got[name]) for name, value in expected.items()} == (
        dict.fromkeys(expected, 0)
    )


def _largest(a: np.ndarray, b: np.ndarray) -> float:
    """The largest difference over the reference's largest magnitude."""
    scale = float(np.max(np.abs(a)))
    return float(np.max(np.abs(a - b))) / scale if scale > 0.0 else 0.0


# ---- The elementwise blocks --------------------------------------------------


def _reference_parts(room: Room, inputs: Inputs) -> dict[str, np.ndarray]:
    """The blocks of a step that call no elementary function, as the reference forms them."""
    model, state, faces, c = (
        room.model,
        inputs.state,
        inputs.faces,
        inputs.conditions,
    )
    dt = model.pseudo_time_step(faces)
    flux_u = np.where(model._live_u, faces.u, 0.0) * model._area_u
    flux_v = np.where(model._live_v, faces.v, 0.0) * model._area_v
    const = model.constants
    g_k = model._conductances(state.nu_t, const.sigma_k)
    g_eps = model._conductances(state.nu_t, const.sigma_e)
    return {
        "dt": dt,
        "strain": model.strain_squared(faces, c),
        "eddy": model.eddy_viscosity(state.k, state.eps),
        "k_advected": model._advect(
            state.k, flux_u, flux_v, c.inflow_k_u, c.inflow_k_v, dt
        ),
        "eps_advected": model._advect(
            state.eps, flux_u, flux_v, c.inflow_eps_u, c.inflow_eps_v, dt
        ),
        "g_u_k": g_k[0],
        "g_v_k": g_k[1],
        "g_u_eps": g_eps[0],
        "g_v_eps": g_eps[1],
    }


def _core_parts(room: Room, inputs: Inputs) -> dict[str, np.ndarray]:
    """The same blocks as the core forms them."""
    core, state, faces, c = room.core, inputs.state, inputs.faces, inputs.conditions
    g_k = core.conductances("k", state.nu_t)
    g_eps = core.conductances("eps", state.nu_t)
    return {
        "dt": core.pseudo_time_step(faces),
        "strain": core.strain_squared(faces, c),
        "eddy": core.eddy_viscosity(state.k, state.eps),
        "k_advected": core.advected(faces, state.k, c.inflow_k_u, c.inflow_k_v),
        "eps_advected": core.advected(faces, state.eps, c.inflow_eps_u, c.inflow_eps_v),
        "g_u_k": g_k[0],
        "g_v_k": g_k[1],
        "g_u_eps": g_eps[0],
        "g_v_eps": g_eps[1],
    }


@pytest.mark.parametrize("flow", FLOWS)
@pytest.mark.parametrize("scheme", SCHEMES)
@pytest.mark.parametrize("name", ROOMS)
def test_the_blocks_equal_the_reference_bit_for_bit(
    name: str, scheme: str, flow: str
) -> None:
    """The step's blocks that call no elementary function (prediction (a))."""
    room = _room(name, STANDARD, scheme)
    inputs = _planted(room, flow, _rng(name, scheme, flow))
    _assert_same(_reference_parts(room, inputs), _core_parts(room, inputs))


def _terms(
    room: Room, inputs: Inputs
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """The growth and decay of a step, the reference's and the core's."""
    args = (inputs.state, inputs.faces, inputs.conditions)
    expected = room.model.terms(*args)
    return {name: getattr(expected, name) for name in TERMS}, room.core.terms(*args)


@pytest.mark.parametrize("flow", FLOWS)
@pytest.mark.parametrize("name", ROOMS)
def test_the_standard_terms_equal_the_reference_bit_for_bit(
    name: str, flow: str
) -> None:
    """Production, growth and decay of the standard variant; R is zero."""
    room = _room(name)
    expected, got = _terms(room, _planted(room, flow, _rng(name, flow)))
    _assert_same(expected, got)
    assert not got["rng_r"].any()


@pytest.mark.parametrize("flow", FLOWS)
@pytest.mark.parametrize("name", ROOMS)
def test_the_rng_terms_equal_the_reference_bit_for_bit(
    name: str, flow: str, libm_numpy: None
) -> None:
    """RNG's R of both signs, its negative part in the growth and its positive in the decay."""
    room = _room(name, RNG)
    expected, got = _terms(room, _planted(room, flow, _rng(name, flow)))
    _assert_same(expected, got)
    if flow == "decades":
        assert np.any(expected["rng_r"] > 0.0) and np.any(expected["rng_r"] < 0.0)


def test_a_production_given_as_a_negative_zero_keeps_its_sign() -> None:
    """The reference's checks admit -0.0, and the growth's sign follows it."""
    room = _room("40x15-uniform-open")
    inputs = _planted(room, "smooth", _rng("zero"))
    production = np.where(inputs.conditions.production_given, -0.0, 1.0)
    conditions = dataclasses.replace(inputs.conditions, production=production)
    expected, got = _terms(room, dataclasses.replace(inputs, conditions=conditions))
    _assert_same(expected, got)
    assert np.any(np.signbit(expected["growth_k"]))


def test_the_rng_terms_are_within_level_one_in_any_environment() -> None:
    """NumPy's own cube moves R by rounding, whatever loops it forms it with."""
    room = _room("80x30", RNG)
    expected, got = _terms(room, _planted(room, "decades", _rng("level one")))
    for name in TERMS:
        assert _largest(expected[name], got[name]) <= FIELD_TOL, name


@pytest.mark.parametrize("flow", FLOWS)
@pytest.mark.parametrize("name", ROOMS)
def test_the_corner_gradients_equal_the_reference_bit_for_bit(
    name: str, flow: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """du/dy and dv/dx at the corners, as the reference forms them inside a call."""
    room = _room(name)
    inputs = _planted(room, flow, _rng(name, flow))
    found = []
    inner = turbulence._corner_difference

    def kept(*args: np.ndarray) -> np.ndarray:
        found.append(inner(*args))
        return found[-1]

    monkeypatch.setattr(turbulence, "_corner_difference", kept)
    room.model.strain_squared(inputs.faces, inputs.conditions)
    c, faces = inputs.conditions, inputs.faces
    got = (
        room.core.corner_gradient(0, faces.u, c.tangential_bottom, c.tangential_top),
        room.core.corner_gradient(1, faces.v, c.tangential_left, c.tangential_right),
    )
    _assert_same(
        {"dudy": found[0], "dvdx": found[1].T},
        dict(zip(("dudy", "dvdx"), got, strict=True)),
    )


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("scheme", SCHEMES)
@pytest.mark.parametrize("vents", [False, True])
@pytest.mark.parametrize("size", SIZES, ids=str)
def test_small_rooms_equal_the_reference_bit_for_bit(
    size: tuple[int, int],
    vents: bool,
    scheme: str,
    variant: str,
    monkeypatch: pytest.MonkeyPatch,
    libm_numpy: None,
) -> None:
    """Planted SOLID cells, edges and vents on every row length: every block and condition."""
    rng = _rng(size, vents, scheme, variant)
    for _ in range(4):
        room = _small_room(size, variant, scheme, rng, vents, monkeypatch)
        for flow in ("decades", "zeros"):
            inputs = _planted(room, flow, rng)
            _assert_same(_reference_parts(room, inputs), _core_parts(room, inputs))
            _assert_same(*_terms(room, inputs))
            _assert_same(*_conditions(room, inputs))
            _assert_same(*_wall_viscosities(room, inputs, rng))
        monkeypatch.undo()


def test_the_vents_faces_carry_a_flux_diffusion_does_not_cross() -> None:
    """The vents' masks on the product room's vents, and that there are some to test."""
    room = _room("80x30-vents-shelf")
    vents = room.boundary.vent_faces()
    for d, (live, inner, normal) in enumerate(
        (
            (room.model._live_u, room.model._inner_u, vents.normal_u),
            (room.model._live_v, room.model._inner_v, vents.normal_v),
        )
    ):
        assert _differ(live.astype(np.uint8), room.core.live_faces(d)) == 0
        assert _differ(inner.astype(np.uint8), room.core.inner_faces(d)) == 0
        assert not inner[normal].any()
    assert vents.any
    kinds = {run.is_inlet for run in room.boundary.vent_runs()}
    assert kinds == {False, True}


@pytest.mark.parametrize("name", ROOMS)
def test_the_fixed_arrays_equal_the_reference(name: str) -> None:
    """The face masks and volumes of the model, and the boundary's walls and inlets."""
    room = _room(name)
    core, walls = room.core, room.core_walls
    expected = {
        "volume": room.model._volume,
        "live_u": room.model._live_u.astype(np.uint8),
        "live_v": room.model._live_v.astype(np.uint8),
        "inner_u": room.model._inner_u.astype(np.uint8),
        "inner_v": room.model._inner_v.astype(np.uint8),
        "wall_cells": room.walls.wall_cells,
    }
    got = {
        "volume": core.volume,
        "live_u": core.live_faces(0),
        "live_v": core.live_faces(1),
        "inner_u": core.inner_faces(0),
        "inner_v": core.inner_faces(1),
        "wall_cells": walls.wall_cells,
    }
    for c, key in enumerate("uv"):
        mask, faces = _reference_stencil(room, key)
        expected[f"mask_{key}"] = mask
        stencil = walls.stencil(c)
        got[f"mask_{key}"] = stencil["mask"]
        order = np.argsort(stencil["corner"])
        for item in ("corner", "cell_low", "cell_high", "distance"):
            expected[f"{item}_{key}"] = faces[item]
            got[f"{item}_{key}"] = stencil[item][order]
    _assert_same(expected, got)
    reference = [
        *(
            face
            for edge in core_backend.WALL_EDGES
            for face in room.walls._inlet_by_edge[edge].values()
        )
    ]
    values = walls.inlet_values()
    assert values[: len(reference)] == [(f.k, f.eps) for f in reference]
    vented = [
        (room.walls._vent_inflow[run.name].k, room.walls._vent_inflow[run.name].eps)
        for run in room.boundary.vent_runs()
        if run.is_inlet
        for _ in run.widths
    ]
    assert values[len(reference) :] == vented


def _reference_stencil(
    room: Room, key: str
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """One component's wall-function faces, by corner, from the reference's own."""
    walls = room.walls._velocity[key]
    _, nx = room.mesh.cell_type.shape
    corner = walls.rows * (nx + 1) + walls.cols
    if key == "u":
        low = walls.frame_t * nx + walls.frame_s - 1
        high = walls.frame_t * nx + walls.frame_s
    else:
        low = (walls.frame_s - 1) * nx + walls.frame_t
        high = walls.frame_s * nx + walls.frame_t
    order = np.argsort(corner)
    return walls.mask.astype(np.uint8), {
        "corner": corner[order].astype(np.int64),
        "cell_low": low[order].astype(np.int64),
        "cell_high": high[order].astype(np.int64),
        "distance": walls.distance[order],
    }


# ---- The wall functions and the conditions -----------------------------------


def _conditions(
    room: Room, inputs: Inputs
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """A step's conditions, the reference's and the core's."""
    expected = room.walls.conditions(inputs.state, inputs.faces)
    return (
        {name: getattr(expected, name) for name in CONDITIONS},
        room.core_walls.conditions(inputs.state, inputs.faces),
    )


def _wall_viscosities(
    room: Room, inputs: Inputs, rng: np.random.Generator
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """``wall_mu`` over planted viscosities elsewhere, the reference's and the core's."""
    ny, nx = room.mesh.cell_type.shape
    elsewhere = {key: rng.random((ny + 1, nx + 1)) for key in ("u", "v")}
    return (
        room.walls.wall_viscosity(inputs.state.k, elsewhere),
        room.core_walls.wall_viscosity(inputs.state.k, elsewhere),
    )


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("flow", FLOWS)
@pytest.mark.parametrize("name", ROOMS)
def test_the_conditions_equal_the_reference_bit_for_bit(
    name: str, flow: str, variant: str, libm_numpy: None
) -> None:
    """The inflow, the tangential velocities, eps held and the production given."""
    room = _room(name, variant)
    inputs = _planted(room, flow, _rng(name, flow, variant))
    expected, got = _conditions(room, inputs)
    _assert_same(expected, got)
    assert got["eps_held"].any()


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("name", ROOMS)
def test_the_wall_viscosity_equals_the_reference_bit_for_bit(
    name: str, variant: str, libm_numpy: None
) -> None:
    """The momentum wall stencil's viscosity, above and below the log law's floor."""
    room = _room(name, variant)
    rng = _rng(name, variant)
    inputs = _planted(room, "smooth", rng)
    _assert_same(*_wall_viscosities(room, inputs, rng))


def test_the_walls_are_within_level_one_in_any_environment() -> None:
    """NumPy's own logarithm and cube move the wall functions by rounding only."""
    room = _room("80x30-vents-shelf")
    rng = _rng("level one")
    inputs = _planted(room, "decades", rng)
    for expected, got in (
        _conditions(room, inputs),
        _wall_viscosities(room, inputs, rng),
    ):
        for name, value in expected.items():
            if value.dtype == np.float64:
                assert _largest(value, got[name]) <= FIELD_TOL, name


def test_a_boundary_used_again_holds_nothing_of_the_call_before() -> None:
    """Two calls on one core boundary: the second is a fresh boundary's."""
    room = _room("80x30-vents-shelf")
    rng = _rng("again")
    first, second = (_planted(room, "decades", rng) for _ in range(2))
    fresh = _build(room.mesh, room.config, room.boundary).core_walls
    room.core_walls.conditions(first.state, first.faces)
    again = room.core_walls.conditions(second.state, second.faces)
    _assert_same(fresh.conditions(second.state, second.faces), again)


def test_a_solid_edge_cell_takes_the_fields_largest_value() -> None:
    """Beside a SOLID edge cell nothing enters, and the largest k or eps stands in."""
    room = _room("80x30")
    inputs = _planted(room, "smooth", _rng("stand in"))
    solid_bottom = room.solid[0, :]
    assert solid_bottom.any()
    got = room.core_walls.conditions(inputs.state, inputs.faces)
    live = ~room.solid
    assert np.all(got["inflow_k_v"][0, solid_bottom] == inputs.state.k[live].max())
    assert np.all(got["inflow_eps_v"][0, solid_bottom] == inputs.state.eps[live].max())


def test_the_largest_value_is_read_over_the_non_solid_cells() -> None:
    """A state whose SOLID edge cell in one corner holds a value, which no check refuses here."""
    room = _room("80x30")
    inputs = _planted(room, "smooth", _rng("solid values"))
    solid = np.argwhere(room.solid)
    state = TurbulenceState(
        **{
            name: _spoiled(getattr(inputs.state, name), 1.0e3, room.solid)
            for name in ("k", "eps", "nu_t")
        }
    )
    # The first SOLID cell lies on the bottom edge and holds its own value; the
    # others there take the largest of the non-SOLID ones.
    assert solid[0][0] == 0 and room.solid[0, solid[0][1] + 1]
    expected = room.walls.conditions(state, inputs.faces)
    got = room.core_walls.conditions(state, inputs.faces)
    names = [n for n in CONDITIONS if n.startswith("inflow")]
    _assert_same({n: getattr(expected, n) for n in names}, {n: got[n] for n in names})
    assert got["inflow_k_v"][0, solid[0][1]] == 1.0e3
    assert got["inflow_k_v"][0, solid[0][1] + 1] < 1.0e3


@pytest.mark.parametrize("quantity", ["k", "eps"])
def test_a_nan_in_the_state_is_the_stand_in_beside_a_solid_edge_cell(
    quantity: str,
) -> None:
    """The boundary does not check the state: a NaN in it stands in where nothing enters.

    The reference's largest value over the cells is NaN when one is, and a SOLID
    edge cell takes it. A run's state is checked before the boundary reads it, so
    only a planted one shows the rule.
    """
    room = _room("80x30")
    inputs = _planted(room, "smooth", _rng("nan stand in", quantity))
    spoiled = _state_with(
        inputs,
        **{quantity: _spoiled(getattr(inputs.state, quantity), np.nan, ~room.solid)},
    )
    with np.errstate(all="ignore"):
        expected = room.walls.conditions(spoiled.state, spoiled.faces)
        got = room.core_walls.conditions(spoiled.state, spoiled.faces)
    names = [n for n in CONDITIONS if n.startswith("inflow")]
    _assert_same({n: getattr(expected, n) for n in names}, {n: got[n] for n in names})
    solid_bottom = room.solid[0, :]
    assert solid_bottom.any()
    assert np.isnan(got[f"inflow_{quantity}_v"][0, solid_bottom]).all()
    other = "eps" if quantity == "k" else "k"
    assert np.isfinite(got[f"inflow_{other}_v"][0, solid_bottom]).all()


@pytest.mark.parametrize("name", [n for n in ROOMS if n != "cavity"])
def test_the_initial_values_equal_the_reference(name: str) -> None:
    """The inflow-weighted means of the inlets' k and eps."""
    room = _room(name)
    assert room.core_walls.initial_values() == room.walls.initial_values()


def test_a_room_without_an_inlet_has_no_initial_values() -> None:
    """A closed cavity: the reference's refusal."""
    room = _room("cavity")
    with pytest.raises(ValueError) as expected:
        room.walls.initial_values()
    with pytest.raises(ValueError) as got:
        room.core_walls.initial_values()
    assert str(got.value) == str(expected.value)


def test_the_initial_values_are_summed_as_python_sums_floats() -> None:
    """Python's compensated ``sum`` over faces whose terms span sixteen decades."""
    mesh = _room("40x15-uniform-open").mesh
    rng = _rng("sum")
    ny, nx = mesh.cell_type.shape
    differs = 0
    for _ in range(300):
        faces = [
            (
                1,
                ny,
                int(i),
                float(rng.uniform(0.01, 0.3)),
                float(rng.choice([-1.0, 1.0]) * 10.0 ** rng.uniform(-8.0, 8.0)),
                float(rng.uniform(0.01, 0.3)),
                float(10.0 ** rng.uniform(-3.0, 3.0)),
            )
            for i in range(nx)
        ]
        walls = _walls_with(mesh, faces)
        values = [inflow_values(f[3], f[4], f[5]) for f in faces]
        inflow = [abs(f[4]) * f[6] for f in faces]
        total = sum(inflow)
        k = sum(v[0] * q for v, q in zip(values, inflow, strict=True)) / total
        eps = sum(v[1] * q for v, q in zip(values, inflow, strict=True)) / total
        assert walls.initial_values() == (k, eps)
        plain = 0.0
        for v, q in zip(values, inflow, strict=True):
            plain += v[0] * q
        differs += plain != sum(v[0] * q for v, q in zip(values, inflow, strict=True))
    # The plain running sum is another number on some, so the rule is tested.
    assert differs > 0


def _walls_with(mesh: Mesh, faces: list[tuple]) -> "_cfdcore.TurbulenceBoundary2":
    """A core boundary of a mesh without walls whose inlets are ``faces``."""
    ny, nx = mesh.cell_type.shape
    lengths = (nx, nx, ny, ny)
    return _cfdcore.TurbulenceBoundary2(
        core_backend.core_grid(mesh),
        [np.zeros(n, dtype=bool) for n in lengths],
        [np.zeros(n + 1, dtype=bool) for n in lengths],
        [(np.ones(n + 1, dtype=bool), np.zeros(n + 1), 0.1) for n in lengths],
        None,
        faces,
        [],
        [],
        0.09,
        1.2,
        1.5e-5,
        core_backend.WALL_LAW,
    )


# ---- The free functions ------------------------------------------------------


def test_log_law_floor_equals_the_reference() -> None:
    """The root of kappa y = ln(E y), on the model's constants and on others."""
    assert _cfdcore.log_law_floor(KAPPA, E_WALL) == Y_STAR_FLOOR
    rng = _rng("floor")
    for _ in range(2000):
        kappa, e = float(rng.uniform(0.2, 0.9)), float(rng.uniform(3.0, 30.0))
        assert _cfdcore.log_law_floor(kappa, e) == log_law_floor(kappa, e)


@pytest.mark.parametrize(("kappa", "e"), [(0.41, 1.0), (1.0, math.e), (0.41, 0.5)])
def test_log_law_floor_refuses_laws_that_do_not_meet(kappa: float, e: float) -> None:
    """The reference's refusal, its message included."""
    with pytest.raises(ValueError) as expected:
        log_law_floor(kappa, e)
    with pytest.raises(ValueError) as got:
        _cfdcore.log_law_floor(kappa, e)
    assert str(got.value) == str(expected.value)


def test_inflow_values_equal_the_reference() -> None:
    """Python's float powers, on both signs of the speed and on zero."""
    rng = _rng("inflow")
    for _ in range(20000):
        args = (
            float(rng.uniform(0.0, 1.0)),
            float(rng.standard_normal() * 10.0 ** rng.uniform(-4.0, 2.0)),
            float(10.0 ** rng.uniform(-3.0, 1.0)),
        )
        assert _cfdcore.inflow_values(*args) == inflow_values(*args)
    assert _cfdcore.inflow_values(0.1, 0.0, 0.1) == inflow_values(0.1, 0.0, 0.1)


@pytest.mark.parametrize("c_mu", [0.09, 0.0845])
def test_wall_viscosity_equals_the_reference_bit_for_bit(
    c_mu: float, libm_numpy: None
) -> None:
    """The scalable wall function's viscosity on both sides of its floor."""
    rng = _rng("wall", c_mu)
    k = 10.0 ** rng.uniform(-9.0, 1.0, 50000)
    y = 10.0 ** rng.uniform(-4.0, -1.0, k.size)
    args = (c_mu, 1.2, 1.5e-5)
    got = _cfdcore.wall_viscosity(k, y, *args, core_backend.WALL_LAW)
    assert _differ(wall_viscosity(k, y, *args), got) == 0
    above = c_mu**0.25 * np.sqrt(k) * y / args[2] > Y_STAR_FLOOR
    assert above.any() and not above.all()


# ---- The step and its solves -------------------------------------------------


@dataclasses.dataclass
class Systems:
    """What the reference's step handed its two implicit solves, and what they returned."""

    calls: list[tuple[tuple, dict]]
    results: list[ImplicitResult]


def _reference_step(
    room: Room, inputs: Inputs, monkeypatch: pytest.MonkeyPatch
) -> tuple[TurbulenceState, Systems]:
    """The reference's step with its implicit solves kept."""
    systems = Systems([], [])

    def kept(*args: object, **kwargs: object) -> ImplicitResult:
        systems.calls.append((args, kwargs))
        systems.results.append(implicit_step(*args, **kwargs))
        return systems.results[-1]

    with monkeypatch.context() as patch:
        patch.setattr(turbulence, "implicit_step", kept)
        state = room.model.step(inputs.state, inputs.faces, inputs.conditions)
    return state, systems


def _core_solve(
    room: Room, args: tuple, held: np.ndarray | None
) -> tuple[np.ndarray, object]:
    """The core's implicit solve on the arguments the reference's step handed its own."""
    out = np.empty_like(args[0])
    grid = core_backend.core_grid(room.mesh)
    result = _cfdcore.implicit_step(
        _cfdcore.ImplicitScratch2(grid),
        args[0],
        args[1],
        args[2],
        args[3],
        args[4],
        args[5],
        args[6],
        args[7],
        args[8],
        out,
        held=held,
        method=CONJUGATE_GRADIENT,
    )
    return out, result


def _assert_step(
    room: Room, inputs: Inputs, monkeypatch: pytest.MonkeyPatch, rule: bool = True
) -> Systems:
    """A step of both sides: the explicit fields bit for bit, the solves by the CG's rule.

    The core's step must also equal the core's solve of the reference's systems
    bit for bit, which ties every operand of its two solves (the held cells, the
    sinks, the steps, the tolerance and the cap) to the reference's. Without
    ``rule`` that is all: a planted system whose condition amplifies a dot
    product's rounding is for the probe to report (prompt 88: near rounding the
    counts can differ).
    """
    expected, systems = _reference_step(room, inputs, monkeypatch)
    args = (inputs.state, inputs.faces, inputs.conditions)
    got = room.core.step(*args)
    k_star, eps_star = room.core.explicit_fields(*args)
    (k_args, _), (eps_args, eps_kwargs) = systems.calls
    _assert_same({"k": k_args[0], "eps": eps_args[0]}, {"k": k_star, "eps": eps_star})
    for i, (name, call, held) in enumerate(
        (("k", k_args, None), ("eps", eps_args, eps_kwargs["held"]))
    ):
        field, result = _core_solve(room, call, held)
        reference = systems.results[i]
        assert _differ(np.where(room.solid, 0.0, field), got[name]) == 0
        assert result.iterations == room.core.last_iterations[i]
        assert result.fallbacks == room.core.last_fallbacks[i]
        if not rule:
            continue
        assert result.iterations == reference.iterations
        assert result.fallbacks == reference.fallbacks
        if reference.fallbacks:
            assert _differ(getattr(expected, name), got[name]) == 0
        else:
            assert _largest(getattr(expected, name), got[name]) <= FIELD_TOL
    if rule:
        assert room.core.solves_converged == room.model.solves_converged
        assert room.core.last_iterations == room.model.last_iterations
        assert room.core.last_fallbacks == room.model.last_fallbacks
    assert _differ(got["nu_t"], room.core.eddy_viscosity(got["k"], got["eps"])) == 0
    return systems


@pytest.mark.parametrize("flow", STEP_FLOWS)
@pytest.mark.parametrize("scheme", SCHEMES)
@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("name", ["80x30", "80x30-vents-shelf", "40x15-uniform-open"])
def test_a_step_equals_the_reference_by_its_blocks(
    name: str,
    variant: str,
    scheme: str,
    flow: str,
    monkeypatch: pytest.MonkeyPatch,
    libm_numpy: None,
) -> None:
    """Explicit fields bit for bit, the solves at the reference's counts within 1e-12."""
    room = _room(name, variant, scheme)
    rng = _rng(name, variant, scheme, flow)
    inputs = _planted(room, flow, rng)
    walls = room.walls.conditions(inputs.state, inputs.faces)
    _assert_step(room, inputs, monkeypatch, rule=False)
    walled = dataclasses.replace(inputs, conditions=walls)
    _assert_step(room, walled, monkeypatch, rule=False)
    for _ in range(3):
        _assert_step(room, _gentle(room, rng), monkeypatch)


@pytest.mark.parametrize("name", ["annex20", "cavity", "160x60-stretched"])
def test_a_step_on_the_other_rooms(
    name: str, monkeypatch: pytest.MonkeyPatch, libm_numpy: None
) -> None:
    """The Annex 20 mesh, a cavity of moving walls and a stretched room."""
    room = _room(name)
    _assert_step(room, _gentle(room, _rng(name)), monkeypatch)


@pytest.mark.parametrize("name", ["80x30", "80x30-vents-shelf"])
def test_advance_is_the_step_with_the_boundarys_conditions(name: str) -> None:
    """One call for the conditions and the step: the two calls' fields."""
    room = _room(name)
    inputs = _planted(room, "smooth", _rng(name))
    conditions = core_backend.turbulence_conditions(
        room.core_walls.conditions(inputs.state, inputs.faces)
    )
    expected = room.core.step(inputs.state, inputs.faces, conditions)
    counts = room.core.last_iterations
    got = room.core.advance(room.core_walls, inputs.state, inputs.faces)
    _assert_same(expected, got)
    assert room.core.last_iterations == counts
    other = core_backend.turbulence_boundary(room.mesh, room.config, room.boundary)
    with pytest.raises(ValueError, match="another grid"):
        room.core.advance(other, inputs.state, inputs.faces)


def _with_solver(
    name: str, tol: float, max_iter: int, cfl: float | None = None, scheme: str = UPWIND
) -> Room:
    """A room whose models solve to ``tol`` in ``max_iter`` at most, at a Courant number."""
    raw = copy.deepcopy(_raw(name))
    raw["turbulence"].update(tol=tol, max_iter=max_iter, advection_scheme=scheme)
    config = SimConfig.from_dict(raw, "configs")
    mesh = Mesh(config)
    room = _build(mesh, config, StaggeredBoundary(mesh, config))
    if cfl is None:
        return room
    # A Courant number above the configuration's bound, so that the explicit
    # advection overshoots below zero.
    object.__setattr__(config.turbulence, "cfl_number", cfl)
    room.model._cfl = cfl
    core = core_backend.k_epsilon(mesh, config, room.boundary)
    return dataclasses.replace(room, core=core)


@pytest.mark.parametrize("max_iter", [1, 2, 5])
def test_a_solve_at_its_cap_falls_back_as_the_reference(
    max_iter: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CG reaches its cap: Jacobi from the explicit field, bit for bit the reference's."""
    room = _with_solver("40x15-uniform-open", 1.0e-13, max_iter)
    systems = _assert_step(
        room, _planted(room, "smooth", _rng("cap", max_iter)), monkeypatch
    )
    assert [r.fallbacks for r in systems.results] == [1, 1]
    assert not room.core.solves_converged


def test_a_cg_iterate_that_is_not_positive_falls_back_as_the_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A loose tolerance ends CG on an iterate with a value at or below zero."""
    room = _with_solver("40x15-uniform-open", 0.1, 500)
    shape = room.mesh.cell_type.shape
    found = [0, 0]
    kept = [0, 0]
    for trial in range(20):
        rng = _rng("negative", trial)
        # Neighbours decades apart under a strong diffusion, and no growth.
        inputs = _planted(room, "smooth", rng)
        inputs = Inputs(
            TurbulenceState(
                k=10.0 ** rng.uniform(-6.0, 2.0, shape),
                eps=10.0 ** rng.uniform(-10.0, 0.0, shape),
                nu_t=10.0 ** rng.uniform(-3.0, 1.0, shape),
            ),
            inputs.faces,
            dataclasses.replace(
                inputs.conditions,
                production_given=np.ones(shape, dtype=bool),
                production=np.zeros(shape),
            ),
        )
        systems = _assert_step(room, inputs, monkeypatch, rule=False)
        for i, (call, held) in enumerate(
            (
                (systems.calls[0][0], None),
                (systems.calls[1][0], systems.calls[1][1]["held"]),
            )
        ):
            reference = systems.results[i]
            # At so loose a tolerance a count can part where a residual lies at the
            # stop; where it does not, the sign's verdict is the reference's.
            if room.core.last_iterations[i] != reference.iterations:
                continue
            field, result = _core_solve(room, call, held)
            assert result.fallbacks == reference.fallbacks
            kept[i] += reference.fallbacks == 0
            if reference.fallbacks:
                # A fallback of a solve well inside its cap is for its sign.
                assert reference.iterations < 500 and result.converged
                assert result.fallback_sweeps == reference.fallback_sweeps
                assert _differ(reference.field, field) == 0
                found[i] += 1
    assert found[0] > 0 and found[1] > 0
    assert kept[0] > 0 and kept[1] > 0


@pytest.mark.parametrize("quantity", ["k", "eps"])
def test_a_positivity_failure_is_the_references(quantity: str) -> None:
    """A Courant number far above one: the explicit field goes negative and so does the step's.

    The other quantity stands at one value carried in everywhere, which a step
    of any size keeps positive, so that the failure is the one named.
    """
    room = _with_solver("40x15-uniform-open", 1.0e-10, 500, cfl=40.0)
    inputs = _planted(room, "smooth", _rng("positivity", quantity))
    shape = room.mesh.cell_type.shape
    ny, nx = shape
    # A uniform flow, without divergence, so that the flat quantity stays flat.
    faces = FaceVelocities.copy_of(np.full((ny, nx + 1), 0.3), np.zeros((ny + 1, nx)))
    inputs = dataclasses.replace(inputs, faces=faces)
    flat = 1.0e-3
    c, s = inputs.conditions, inputs.state
    other = "eps" if quantity == "k" else "k"
    state = TurbulenceState(
        **{
            other: np.full(shape, flat),
            quantity: getattr(s, quantity),
            "nu_t": np.zeros(shape),
        }
    )
    conditions = dataclasses.replace(
        c,
        **{
            f"inflow_{other}_u": np.full((ny, nx + 1), flat),
            f"inflow_{other}_v": np.full((ny + 1, nx), flat),
        },
        eps_held=np.zeros(shape, dtype=bool),
        production_given=np.ones(shape, dtype=bool),
        production=np.zeros(shape),
    )
    with pytest.raises(PositivityError) as expected:
        room.model.step(state, inputs.faces, conditions)
    with pytest.raises(_cfdcore.PositivityFailure) as got:
        room.core.step(state, inputs.faces, conditions)
    error = core_backend.positivity_error(got.value)
    assert str(error).startswith(f"{quantity} is not positive")
    if room.model.last_fallbacks[0 if quantity == "k" else 1]:
        assert str(error) == str(expected.value)
        assert error.minimum == expected.value.minimum
    else:
        assert error.minimum == pytest.approx(expected.value.minimum, rel=1e-9)
        assert str(error).split("(")[0] == str(expected.value).split("(")[0]
    assert room.core.last_iterations == room.model.last_iterations
    assert room.core.last_fallbacks == room.model.last_fallbacks


def test_a_positivity_failure_by_infinity_is_the_references() -> None:
    """A field that overflows: infinite values pass `> 0` and are refused."""
    room = _with_solver("40x15-uniform-open", 1.0e-10, 3)
    inputs = _planted(room, "smooth", _rng("infinite"))
    production = np.where(room.solid, 0.0, 1.0e308)
    conditions = dataclasses.replace(
        inputs.conditions,
        production_given=~room.solid,
        production=production,
    )
    with np.errstate(all="ignore"), pytest.raises(PositivityError) as expected:
        room.model.step(inputs.state, inputs.faces, conditions)
    with pytest.raises(_cfdcore.PositivityFailure) as got:
        room.core.step(inputs.state, inputs.faces, conditions)
    error = core_backend.positivity_error(got.value)
    assert str(error) == str(expected.value)
    assert np.array_equal(error.minimum, expected.value.minimum, equal_nan=True)


# ---- The refusals ------------------------------------------------------------


def _spoiled(a: np.ndarray, value: float, where: np.ndarray) -> np.ndarray:
    """A copy with one value planted at the first cell the mask names."""
    out = np.array(a)
    out[tuple(np.argwhere(where)[0])] = value
    return out


def _state_with(inputs: Inputs, **arrays: np.ndarray) -> Inputs:
    return dataclasses.replace(
        inputs, state=dataclasses.replace(inputs.state, **arrays)
    )


def _conditions_with(inputs: Inputs, **arrays: np.ndarray) -> Inputs:
    return dataclasses.replace(
        inputs, conditions=dataclasses.replace(inputs.conditions, **arrays)
    )


def _faces_with(inputs: Inputs, u: np.ndarray, v: np.ndarray) -> Inputs:
    return dataclasses.replace(inputs, faces=FaceVelocities.copy_of(u, v))


def _reference_refusals() -> dict[str, Callable[[Room, Inputs], Inputs]]:
    """What makes a step's arguments ones the reference refuses, by name."""
    cases: dict[str, Callable[[Room, Inputs], Inputs]] = {}
    for name in ("k", "eps", "nu_t"):
        cases[f"state-{name}-shape"] = lambda r, i, n=name: _state_with(
            i, **{n: getattr(i.state, n)[:-1]}
        )
        cases[f"state-{name}-dimension"] = lambda r, i, n=name: _state_with(
            i, **{n: getattr(i.state, n).ravel()}
        )
        for label, value in (("negative", -1.0), ("nan", np.nan), ("inf", np.inf)):
            cases[f"state-{name}-{label}"] = lambda r, i, n=name, x=value: _state_with(
                i, **{n: _spoiled(getattr(i.state, n), x, ~r.solid)}
            )
        cases[f"state-{name}-solid"] = lambda r, i, n=name: _state_with(
            i, **{n: _spoiled(getattr(i.state, n), 1.0, r.solid)}
        )
    for name in ("k", "eps"):
        cases[f"state-{name}-zero"] = lambda r, i, n=name: _state_with(
            i, **{n: _spoiled(getattr(i.state, n), 0.0, ~r.solid)}
        )
    cases["faces-shape"] = lambda r, i: _faces_with(
        i, i.faces.u[:, :-1], i.faces.v[:, :-1]
    )
    cases["faces-u-nan"] = lambda r, i: _faces_with(
        i, _spoiled(i.faces.u, np.nan, np.ones_like(i.faces.u, bool)), i.faces.v
    )
    cases["faces-v-inf"] = lambda r, i: _faces_with(
        i, i.faces.u, _spoiled(i.faces.v, -np.inf, np.ones_like(i.faces.v, bool))
    )
    cases["nothing-moves"] = lambda r, i: _faces_with(
        i, np.zeros_like(i.faces.u), np.zeros_like(i.faces.v)
    )
    floats = [n for n in CONDITIONS if n not in ("eps_held", "production_given")]
    for name in floats:
        cases[f"conditions-{name}-shape"] = lambda r, i, n=name: _conditions_with(
            i, **{n: getattr(i.conditions, n)[:-1]}
        )
        cases[f"conditions-{name}-dtype"] = lambda r, i, n=name: _conditions_with(
            i, **{n: getattr(i.conditions, n).astype(np.float32)}
        )
        cases[f"conditions-{name}-nan"] = lambda r, i, n=name: _conditions_with(
            i,
            **{
                n: _spoiled(
                    getattr(i.conditions, n),
                    np.nan,
                    np.ones_like(getattr(i.conditions, n), bool),
                )
            },
        )
    for name in ("eps_held", "production_given"):
        cases[f"conditions-{name}-shape"] = lambda r, i, n=name: _conditions_with(
            i, **{n: getattr(i.conditions, n)[:-1]}
        )
        cases[f"conditions-{name}-dtype"] = lambda r, i, n=name: _conditions_with(
            i, **{n: getattr(i.conditions, n).astype(np.int64)}
        )
        cases[f"conditions-{name}-solid"] = lambda r, i, n=name: _conditions_with(
            i, **{n: _spoiled(getattr(i.conditions, n), True, r.solid)}
        )
    edges = {"low": (0, 1), "high": (-1, -2)}
    for name in ("inflow_k_u", "inflow_eps_u", "inflow_k_v", "inflow_eps_v"):
        for label, (edge, _) in edges.items():
            cases[f"conditions-{name}-{label}"] = lambda r, i, n=name, e=edge: (
                _conditions_with(
                    i, **{n: _edge_spoiled(getattr(i.conditions, n), n, e)}
                )
            )
        cases[f"conditions-{name}-vent"] = lambda r, i, n=name: _conditions_with(
            i,
            **{
                n: np.where(
                    r.boundary.vent_faces().normal_u
                    if n.endswith("u")
                    else r.boundary.vent_faces().normal_v,
                    0.0,
                    getattr(i.conditions, n),
                )
            },
        )
    cases["conditions-eps-wall"] = lambda r, i: _conditions_with(
        i, eps_wall=np.where(i.conditions.eps_held, 0.0, 1.0)
    )
    cases["conditions-production"] = lambda r, i: _conditions_with(
        i, production=np.where(i.conditions.production_given, -1.0, 1.0)
    )
    return cases


def _edge_spoiled(a: np.ndarray, name: str, edge: int) -> np.ndarray:
    """An inflow array with a zero in the middle of one domain edge."""
    out = np.array(a)
    middle = out.shape[0] // 2 if name.endswith("u") else out.shape[1] // 2
    out[(middle, edge) if name.endswith("u") else (edge, middle)] = 0.0
    return out


REFUSALS = _reference_refusals()


@pytest.mark.parametrize("case", list(REFUSALS))
def test_a_step_refuses_what_the_reference_refuses(case: str) -> None:
    """The reference's checks with its types and messages, in its order."""
    room = _room("80x30-vents-shelf")
    inputs = REFUSALS[case](room, _planted(room, "smooth", _rng(case)))
    args = (inputs.state, inputs.faces, inputs.conditions)
    with pytest.raises(ValueError) as expected:
        room.model.step(*args)
    with pytest.raises(ValueError) as got:
        room.core.step(*args)
    assert type(got.value) is type(expected.value)
    assert str(got.value) == str(expected.value)


def test_the_refusals_of_two_arguments_come_in_the_references_order() -> None:
    """The state before the faces, the faces before the conditions, these before the step."""
    room = _room("80x30-vents-shelf")
    inputs = _planted(room, "smooth", _rng("order"))
    bad = ["state-nu_t-negative", "faces-u-nan", "conditions-inflow_k_v-nan"]
    for count in range(1, len(bad) + 1):
        spoiled = inputs
        for case in bad[-count:]:
            spoiled = REFUSALS[case](room, spoiled)
        args = (spoiled.state, spoiled.faces, spoiled.conditions)
        with pytest.raises(ValueError) as expected:
            room.model.step(*args)
        with pytest.raises(ValueError) as got:
            room.core.step(*args)
        assert str(got.value) == str(expected.value)
        # And before the refusal of a true-time step.
        with pytest.raises(ValueError) as got:
            room.core.step(*args, dt=0.1)
        assert str(got.value) == str(expected.value)


ADVANCE_CASES = [
    case
    for case in REFUSALS
    if case.startswith(("state-", "faces-")) or case == "nothing-moves"
]


@pytest.mark.parametrize("case", ADVANCE_CASES)
def test_advance_refuses_the_state_and_the_faces_as_the_reference_does(
    case: str,
) -> None:
    """`advance` reads its state and its faces as `step` does, with the same types and messages."""
    room = _room("80x30-vents-shelf")
    inputs = REFUSALS[case](room, _planted(room, "smooth", _rng("advance", case)))
    with pytest.raises(ValueError) as expected:
        room.model.step(inputs.state, inputs.faces, inputs.conditions)
    with pytest.raises(ValueError) as got:
        room.core.advance(room.core_walls, inputs.state, inputs.faces)
    assert type(got.value) is type(expected.value)
    assert str(got.value) == str(expected.value)


def test_advance_refuses_the_state_before_the_faces() -> None:
    """The reference's order: a state that is refused is named, not the faces beside it."""
    room = _room("80x30-vents-shelf")
    inputs = _planted(room, "smooth", _rng("advance order"))
    bad = REFUSALS["faces-u-nan"](room, REFUSALS["state-nu_t-negative"](room, inputs))
    with pytest.raises(ValueError) as expected:
        room.model.step(bad.state, bad.faces, bad.conditions)
    with pytest.raises(ValueError) as got:
        room.core.advance(room.core_walls, bad.state, bad.faces)
    assert "state.nu_t" in str(got.value)
    assert str(got.value) == str(expected.value)


def _uniform_state(room: Room, k: float, eps: float) -> TurbulenceState:
    """A state of one k and one eps in every non-SOLID cell and no eddy viscosity."""
    shape = room.mesh.cell_type.shape
    return TurbulenceState(
        k=np.where(room.solid, 0.0, k),
        eps=np.where(room.solid, 0.0, eps),
        nu_t=np.zeros(shape),
    )


FORMED = {
    # The wall cells' eps is a cube of the friction velocity: it underflows to zero.
    "eps-wall-underflows": (1.0e-220, 1.0e-10),
    # And overflows to infinity, which the reference's boundary does not check either.
    "eps-wall-overflows": (1.0e208, 1.0e-10),
}


@pytest.mark.parametrize("case", list(FORMED))
def test_advance_refuses_the_conditions_it_forms_as_the_reference_does(
    case: str,
) -> None:
    """The step's check of its conditions, made whoever formed them (review 94, S4).

    Without it a state whose wall values underflow is a ``PositivityFailure`` of a
    step the reference refuses with a ``ValueError`` before any arithmetic.
    """
    room = _room("80x30")
    inputs = _planted(room, "smooth", _rng("formed", case))
    state = _uniform_state(room, *FORMED[case])
    with np.errstate(all="ignore"):
        formed = room.walls.conditions(state, inputs.faces)
        with pytest.raises(ValueError) as expected:
            room.model.step(state, inputs.faces, formed)
        with pytest.raises(ValueError) as got:
            room.core.advance(room.core_walls, state, inputs.faces)
    assert type(got.value) is type(expected.value)
    assert str(got.value) == str(expected.value)
    assert "eps_wall" in str(got.value)


def _zero_intensity_room(name: str, vent: bool) -> Room:
    """A room whose inlets, or whose discharge vents, carry in k and eps of zero.

    ``SimConfig`` allows no turbulence intensity of zero, so the segments are
    changed after it has validated them; the core's boundary accepts the number.
    """
    config = SimConfig.from_dict(copy.deepcopy(_raw(name)), "configs")
    mesh = Mesh(config)
    if vent:
        segments = [
            run.name
            for run in StaggeredBoundary(mesh, config).vent_runs()
            if run.is_inlet
        ]
    else:
        segments = ["hepa_supply"]
    assert segments
    for segment in segments:
        object.__setattr__(config.boundaries[segment], "turbulence_intensity", 0.0)
    return _build(mesh, config, StaggeredBoundary(mesh, config))


@pytest.mark.parametrize("vent", [False, True])
def test_advance_refuses_an_inflow_that_is_not_positive_as_step_does(
    vent: bool,
) -> None:
    """An inlet of no turbulence intensity carries in k and eps of zero.

    The core's boundary accepts the intensity, as ``SimConfig`` does not; the
    reference's step refuses the conditions it forms, on a domain face or on a
    vent's, and so does ``advance`` (review 94, S4).
    """
    room = _zero_intensity_room("80x30-vents-shelf" if vent else "80x30", vent)
    inputs = _planted(room, "smooth", _rng("no intensity", vent))
    formed = core_backend.turbulence_conditions(
        room.core_walls.conditions(inputs.state, inputs.faces)
    )
    args = (inputs.state, inputs.faces, formed)
    with pytest.raises(ValueError) as expected:
        room.model.step(*args)
    with pytest.raises(ValueError) as stepped:
        room.core.step(*args)
    with pytest.raises(ValueError) as advanced:
        room.core.advance(room.core_walls, inputs.state, inputs.faces)
    where = "vent" if vent else "domain"
    message = f"conditions' inflow k and eps must be positive on the {where} faces"
    assert str(expected.value) == str(stepped.value) == message
    assert type(advanced.value) is type(expected.value)
    assert str(advanced.value) == message


def _records(
    model: "KEpsilonModel | _cfdcore.KEpsilon2",
) -> tuple[tuple[int, int], tuple[int, int], bool]:
    """The counts and the verdict a step leaves on a model, the reference's or the core's."""
    return model.last_iterations, model.last_fallbacks, model.solves_converged


def test_a_refused_call_leaves_the_records_of_the_step_before() -> None:
    """The counts stand through a call that is refused, as the reference's do.

    The refusals here are the ones made before the solves: of the state, of the
    faces, of the conditions and of a field at rest. A step that fails its
    positivity check, by contrast, has set them (the reference does as well).
    """
    room = _with_solver("40x15-uniform-open", 1.0e-10, 500)
    inputs = _planted(room, "smooth", _rng("records"))
    args = (inputs.state, inputs.faces, inputs.conditions)
    room.model.step(*args)
    room.core.step(*args)
    kept = _records(room.core)
    assert kept == _records(room.model)
    assert kept[0] != (0, 0)
    rest = REFUSALS["nothing-moves"](room, inputs)
    calls = {
        "step at rest": lambda: room.core.step(rest.state, rest.faces, rest.conditions),
        "advance at rest": lambda: room.core.advance(
            room.core_walls, rest.state, rest.faces
        ),
    }
    for case in ("state-k-negative", "faces-u-nan", "conditions-eps-wall"):
        bad = REFUSALS[case](room, inputs)
        calls[case] = lambda b=bad: room.core.step(b.state, b.faces, b.conditions)
    for name, call in calls.items():
        with pytest.raises(ValueError):
            call()
        assert _records(room.core) == kept, name
    with pytest.raises(ValueError):
        room.model.step(rest.state, rest.faces, rest.conditions)
    assert _records(room.model) == kept


@pytest.mark.parametrize("dt", [1.0e-3, 1, True, np.float64(0.1), "a"])
def test_a_true_time_step_is_refused(dt: object) -> None:
    """The true-time step of VAL-015 stays on the reference (ADR-017 F)."""
    room = _room("40x15-uniform-open")
    inputs = _planted(room, "smooth", _rng("true time"))
    with pytest.raises(NotImplementedError, match="true-time"):
        room.core.step(inputs.state, inputs.faces, inputs.conditions, dt)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (_strided, "C-contiguous"),
        (_fortran, "C-contiguous"),
        (lambda a: a.astype(np.float32), "C-contiguous float64"),
        (_unaligned_like, "not aligned"),
    ],
    ids=["strided", "fortran", "float32", "unaligned"],
)
@pytest.mark.parametrize("name", ["k", "eps", "nu_t"])
def test_an_array_the_core_cannot_read_is_a_type_error(
    name: str, change: Callable[[np.ndarray], np.ndarray], message: str
) -> None:
    """Layouts and dtypes the reference reads and the core does not."""
    room = _room("40x15-uniform-open")
    inputs = _planted(room, "smooth", _rng("layout"))
    spoiled = _state_with(inputs, **{name: change(getattr(inputs.state, name))})
    with pytest.raises(TypeError, match=message):
        room.core.step(spoiled.state, spoiled.faces, spoiled.conditions)


def test_the_state_is_the_references() -> None:
    """``state``: copies zero in SOLID cells, and their eddy viscosity."""
    room = _room("80x30")
    rng = _rng("state")
    shape = room.mesh.cell_type.shape
    k, eps = 10.0 ** rng.uniform(-6, 0, shape), 10.0 ** rng.uniform(-6, 0, shape)
    expected = room.model.state(k, eps)
    got = room.core.state(k, eps)
    _assert_same({n: getattr(expected, n) for n in ("k", "eps", "nu_t")}, got)
    for spoil in (
        (k[:-1], eps),
        (k, eps[:, :-1].copy()),
        (_spoiled(k, 0.0, ~room.solid), eps),
        (k, _spoiled(eps, np.inf, ~room.solid)),
    ):
        with pytest.raises(ValueError) as wanted:
            room.model.state(*spoil)
        with pytest.raises(ValueError) as found:
            room.core.state(*spoil)
        assert str(found.value) == str(wanted.value)


def _model_arguments(**changes: object) -> list[object]:
    """KEpsilon2's arguments for a small room, some replaced."""
    mesh = _room("40x15-uniform-open").mesh
    args = {
        "grid": core_backend.core_grid(mesh),
        "dx_cell": mesh.dx_cell,
        "dy_cell": mesh.dy_cell,
        "dx_face": mesh.dx_face,
        "dy_face": mesh.dy_face,
        "variant": (0.09, 1.44, 1.92, 1.0, 1.3, None, None),
        "cfl": 0.25,
        "upwind": True,
        "max_iter": 500,
        "tol": 1.0e-10,
        "nu": 1.5e-5,
        "vent_u": None,
        "vent_v": None,
    }
    args.update(changes)
    return list(args.values())


@pytest.mark.parametrize(
    ("changes", "error", "message"),
    [
        ({"dx_cell": np.ones(3)}, ValueError, "cell widths"),
        ({"dy_face": np.ones(3)}, ValueError, "face widths"),
        ({"dx_face": np.zeros(41)}, ValueError, "width is not finite"),
        ({"variant": (0.09, 1.44)}, TypeError, "variant is"),
        (
            {"variant": (0.0, 1.44, 1.92, 1.0, 1.3, None, None)},
            ValueError,
            "constant of the variant",
        ),
        (
            {"variant": (0.09, 1.44, 1.92, 1.0, 1.3, -4.0, 0.01)},
            ValueError,
            "eta_0 or beta",
        ),
        (
            {"variant": (True, 1.44, 1.92, 1.0, 1.3, None, None)},
            TypeError,
            "c_mu must be a number",
        ),
        ({"cfl": 0.0}, ValueError, "cfl or nu"),
        ({"cfl": True}, TypeError, "cfl must be a number"),
        ({"upwind": 1}, TypeError, "incompatible function arguments"),
        ({"max_iter": 2.5}, TypeError, "max_iter must be an int"),
        ({"max_iter": True}, TypeError, "max_iter must be an int"),
        ({"max_iter": 0}, ValueError, "max_iter is not positive"),
        ({"max_iter": -3}, ValueError, "max_iter is not positive"),
        ({"tol": "a"}, TypeError, "tol must be a number"),
        ({"tol": True}, TypeError, "tol must be a number"),
        ({"tol": 0.0}, ValueError, "tol is not finite and positive"),
        ({"tol": -1.0e-10}, ValueError, "tol is not finite and positive"),
        ({"tol": math.nan}, ValueError, "tol is not finite and positive"),
        ({"tol": math.inf}, ValueError, "tol is not finite and positive"),
        ({"nu": math.inf}, ValueError, "cfl or nu"),
        (
            {"vent_u": np.zeros((15, 41), dtype=bool)},
            ValueError,
            "some directions",
        ),
        ({"vent_v": np.zeros((15, 41), dtype=bool)}, TypeError, "vent_v is not"),
    ],
)
def test_the_models_construction_refuses(
    changes: dict[str, object], error: type[Exception], message: str
) -> None:
    """Arguments a model cannot be made from, each refused for its own reason."""
    _cfdcore.KEpsilon2(*_model_arguments())
    with pytest.raises(error, match=message):
        _cfdcore.KEpsilon2(*_model_arguments(**changes))


def _walls_arguments(**changes: object) -> list[object]:
    """TurbulenceBoundary2's arguments for a small room, some replaced."""
    mesh = _room("40x15-uniform-open").mesh
    ny, nx = mesh.cell_type.shape
    lengths = (nx, nx, ny, ny)
    args = {
        "grid": core_backend.core_grid(mesh),
        "wall_cells": [np.zeros(n, dtype=bool) for n in lengths],
        "wall_corners": [np.zeros(n + 1, dtype=bool) for n in lengths],
        "tangential": [
            (np.ones(n + 1, dtype=bool), np.zeros(n + 1), 0.1) for n in lengths
        ],
        "vents": None,
        "edge_inlets": [(1, ny, 3, 0.05, -0.4, 0.1, 0.2)],
        "vent_inlets": [],
        "exhausts": [],
        "c_mu": 0.09,
        "rho": 1.2,
        "nu": 1.5e-5,
        "law": core_backend.WALL_LAW,
    }
    args.update(changes)
    return list(args.values())


def _spoil_edge(n: int, where: int) -> list[np.ndarray]:
    lengths = [40, 40, 15, 15]
    lengths[where] = n
    return [np.zeros(m, dtype=bool) for m in lengths]


VENTS_4 = (
    np.zeros((15, 41), dtype=bool),
    np.zeros((16, 40), dtype=bool),
    np.zeros((16, 41), dtype=bool),
    np.zeros((16, 41), dtype=bool),
)


@pytest.mark.parametrize(
    ("changes", "error", "message"),
    [
        (
            {"wall_cells": _spoil_edge(40, 0)[:3]},
            TypeError,
            "bottom, top, left and right",
        ),
        ({"wall_cells": _spoil_edge(39, 0)}, TypeError, "bottom wall cells"),
        ({"wall_cells": _spoil_edge(16, 3)}, TypeError, "right wall cells"),
        ({"wall_corners": _spoil_edge(40, 1)}, TypeError, "bottom wall corners"),
        (
            {"tangential": [(np.ones(41, dtype=bool), np.zeros(41))] * 4},
            TypeError,
            "wall_distance",
        ),
        ({"vents": VENTS_4[:3]}, TypeError, "normal_u, normal_v"),
        ({"vents": (VENTS_4[1], *VENTS_4[1:])}, TypeError, "vent mask is not"),
        (
            {"vents": (VENTS_4[0].astype(float), *VENTS_4[1:])},
            TypeError,
            "bool ndarray",
        ),
        (
            {"edge_inlets": [(1, 15, 3, 0.05, -0.4, 0.1)]},
            TypeError,
            "an inlet's face is",
        ),
        (
            {"edge_inlets": [(1, 15, 40, 0.05, -0.4, 0.1, 0.2)]},
            ValueError,
            "not on the grid",
        ),
        (
            {"edge_inlets": [(2, 15, 3, 0.05, -0.4, 0.1, 0.2)]},
            ValueError,
            "direction is 0",
        ),
        (
            {"vent_inlets": [(1, 15, 3, math.nan, -0.4, 0.1, 0.2)]},
            ValueError,
            "inlet's number",
        ),
        (
            {"vent_inlets": [(1, 15, 3, 0.05, -0.4, 0.0, 0.2)]},
            ValueError,
            "inlet's number",
        ),
        (
            {"edge_inlets": [(1, 15, 3, True, -0.4, 0.1, 0.2)]},
            TypeError,
            "intensity must be a number",
        ),
        ({"exhausts": [(0, 3)]}, TypeError, "an exhaust's face is"),
        ({"exhausts": [(0, 3, 0)]}, ValueError, "domain edge"),
        ({"exhausts": [(1, 15, 2)]}, ValueError, "domain edge"),
        ({"c_mu": 0.0}, ValueError, "constant is not"),
        ({"nu": True}, TypeError, "nu must be a number"),
        ({"law": (0.41, 9.8)}, TypeError, "law of the wall"),
        ({"law": (0.41, 9.8, -1.0)}, ValueError, "constant is not"),
    ],
)
def test_the_boundarys_construction_refuses(
    changes: dict[str, object], error: type[Exception], message: str
) -> None:
    """Arguments a boundary cannot be made from, each refused for its own reason."""
    _cfdcore.TurbulenceBoundary2(*_walls_arguments())
    with pytest.raises(error, match=message):
        _cfdcore.TurbulenceBoundary2(*_walls_arguments(**changes))


def test_an_inlet_without_turbulence_is_refused_as_the_reference_refuses_it() -> None:
    """The mapping's refusal, which names the segment (the core holds no names)."""
    room = _room("80x30")
    config = SimConfig.from_dict(_raw("80x30"), "configs")
    object.__setattr__(config.boundaries["hepa_supply"], "turbulence_intensity", None)
    with pytest.raises(ValueError) as expected:
        TurbulenceBoundary(room.mesh, config, room.boundary)
    with pytest.raises(ValueError) as got:
        core_backend.turbulence_boundary(room.mesh, config, room.boundary)
    assert str(got.value) == str(expected.value)


def test_the_references_vents_are_the_boundarys() -> None:
    """The runs the reference's model derives are the boundary's, which the core is handed."""
    room = _room("80x30-vents")
    faces = room.boundary.vent_faces()
    runs = vent_runs(room.mesh, room.config, room.boundary._registry)
    derived = vent_face_masks(room.mesh, runs)
    assert _differ(derived.normal_u, faces.normal_u) == 0
    assert _differ(derived.normal_v, faces.normal_v) == 0


# ---- The solver's writes and the positivity test (prompt 95, section 0) -------


def test_the_least_value_of_a_failure_is_read_over_the_non_solid_cells() -> None:
    """One infinity and no NaN: the least value is the smallest positive one, not a zero.

    A step reaches no such field (prompt 95, section 0: an overflow in a step came
    with a NaN beside it, so the least value is NaN whichever cells are read), so
    the test is the check given the field, the reference's ``_check_positive``. The
    SOLID cells hold the zeros a step leaves there, which a least value read over
    them too would return.
    """
    room = _room("80x30-vents-shelf")
    assert room.solid.any()
    rng = _rng("least over the non-SOLID cells")
    q = np.where(room.solid, 0.0, 0.5 + rng.random(room.solid.shape))
    j, i = map(int, np.argwhere(~room.solid)[17])
    q[j, i] = math.inf
    for name in ("k", "eps"):
        with pytest.raises(PositivityError) as expected:
            room.model._check_positive(q, name)
        with pytest.raises(_cfdcore.PositivityFailure) as got:
            room.core.check_positive(name, q)
        error = core_backend.positivity_error(got.value)
        assert str(error) == str(expected.value)
        assert error.minimum == expected.value.minimum
        assert 0.5 <= error.minimum < 1.5
    q[j, i] = 2.0
    room.model._check_positive(q, "k")
    assert room.core.check_positive("k", q) is None


@pytest.mark.parametrize("name", ["80x30", "80x30-vents-shelf", "annex20"])
def test_the_relaxation_is_the_solvers_bit_for_bit(name: str) -> None:
    """``_turbulence_step``'s relaxed nu_t, from the step's own nu_t and the old one."""
    from src.solver_staggered import StaggeredSolver

    room = _room(name)
    solver = StaggeredSolver(room.mesh, room.config, room.boundary)
    alpha = room.config.turbulence.alpha_turbulence
    assert alpha not in (0.5, 1.0)
    inputs = _gentle(room, _rng("relaxation", name))
    stepped = room.model.step(
        inputs.state, inputs.faces, room.walls.conditions(inputs.state, inputs.faces)
    )
    relaxed = solver._turbulence_step(
        inputs.state, np.array(inputs.faces.u), np.array(inputs.faces.v), 0
    )
    got = _cfdcore.relax_eddy_viscosity(inputs.state.nu_t, stepped.nu_t, alpha)
    assert _differ(got, relaxed.nu_t) == 0
    assert _differ(got, stepped.nu_t) > 0 and _differ(got, inputs.state.nu_t) > 0


def test_the_outlets_extrapolation_is_the_solvers_on_every_edge() -> None:
    """``_extrapolate_outlets`` on the small rooms, which hold a pressure outlet on each edge."""
    from src.solver_staggered import StaggeredSolver

    held = dict.fromkeys(("left", "right", "bottom", "top"), 0)
    for nx, ny in SIZES:
        config = SimConfig.from_dict(
            _small_raw(nx, ny, 1.05, STANDARD, UPWIND), "configs"
        )
        mesh = Mesh(config)
        boundary = StaggeredBoundary(mesh, config)
        solver = StaggeredSolver(mesh, config, boundary)
        rng = _rng("outlets", nx, ny)
        u = rng.standard_normal((ny, nx + 1))
        v = rng.standard_normal((ny + 1, nx))
        core = core_backend.outlet_faces(mesh, boundary)
        got_u, got_v = core.extrapolate(u, v)
        solver._extrapolate_outlets(u, v)
        assert _differ(got_u, u) == 0 and _differ(got_v, v) == 0
        outlets = boundary.pressure_outlets()
        for edge in held:
            held[edge] += int(outlets[edge].is_outlet.sum())
        assert core.count == sum(int(o.is_outlet.sum()) for o in outlets.values())
    assert all(count > 0 for count in held.values())
