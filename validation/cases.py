"""Load validation case configurations from the committed YAML files.

Each validation case has exactly one configuration, checked in under
configs/, so the tests and the benchmark harness run the same solve from
the same parameters. The overrides this module makes are the grid, which is
the one axis a refinement study varies; the wall clustering ECR-001
acceptance criterion 2 fixes, which load_wall_clustered derives from the
case's own height and cell count rather than from a second file; the Annex
20 inlet velocity, which load_annex20 scales so the faces the slot covers
carry the specified flux on the grid at hand (ADR-012 G (iii)); and the
stopping rule, which with_velocity_step pins to velocity_step for a solve
that must reproduce a field saved under it; and the backward-facing step's
y clustering, which load_backstep derives so that a face falls on the step
height on any grid (ADR-012 G (ii)). A named preset is loaded by
load_preset, which picks the loader, so no consumer repeats that choice.
"""

import copy
from pathlib import Path

import numpy as np
import yaml

from src.boundary_staggered import StaggeredBoundary
from src.config import PRESSURE_PRECONDITIONERS, VELOCITY_STEP, SimConfig
from src.mesh import Mesh, geometric_faces, ratio_for_wall_spacing

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"

CASE_FILES: dict[str, str] = {
    "poiseuille": "validation_poiseuille.yaml",
    "cavity": "validation_cavity.yaml",
    "annex20": "validation_annex20.yaml",
    "backstep": "validation_backstep.yaml",
}

# Named grid presets: identifier -> (case, nx, ny). The tuple does not say
# whether a preset is wall-clustered, so a preset is loaded by load_preset and
# never from its tuple. The harness accepts every identifier; the field viewer
# accepts the uniform ones, so a benchmark row and a picture of the same solve
# are named the same way.
CASE_GRIDS: dict[str, tuple[str, int, int]] = {
    "val001_80x40": ("poiseuille", 80, 40),
    "val001_80x40_stretched": ("poiseuille", 80, 40),
    "val002_20x20": ("cavity", 20, 20),
    "val002_40x40": ("cavity", 40, 40),
    "val002_80x80": ("cavity", 80, 80),
    "val017_216x72": ("annex20", 216, 72),
    "val017_375x125": ("annex20", 375, 125),
    "val019_410x40": ("backstep", 410, 40),
    "val019_656x60": ("backstep", 656, 60),
    "val019_984x90": ("backstep", 984, 90),
}

# Presets loaded by load_wall_clustered; every other preset takes its whole
# mesh from the case file.
WALL_CLUSTERED_GRIDS: frozenset[str] = frozenset({"val001_80x40_stretched"})

# Presets loaded by load_annex20, which scales the inlet velocity to the grid.
ANNEX20_GRIDS: frozenset[str] = frozenset({"val017_216x72", "val017_375x125"})

# The Annex 20 case's inlet segment, whose velocity load_annex20 scales.
ANNEX20_INLET = "inlet"

# Presets loaded by load_backstep, which clusters y so a face falls on the
# step height.
BACKSTEP_GRIDS: frozenset[str] = frozenset(
    {"val019_410x40", "val019_656x60", "val019_984x90"}
)

# The backward-facing step's obstacle, whose top edge is the step height and
# whose right edge is the step's face at x = 0 of the case's frame.
BACKSTEP_OBSTACLE = "step"

# The step height's face is the one this share of ny up from the floor, so
# the clustered cells under the step height are 1.5 times as many as a uniform
# spacing would put there (ADR-012 G (ii): the step's corner on a grid line).
STEP_FACE_SHARE = 1.0 / 6.0

# Relative tolerance on a face coordinate that must coincide with an edge of
# the geometry: far above the rounding of a face built by cumulative sums,
# far below any cell width.
FACE_TOLERANCE = 1e-9

# ECR-001 criterion 2, amended 2026-09-24: the wall-adjacent cell is this
# fraction of the uniform spacing H / ny, and the geometric ratio is derived.
WALL_SPACING_FRACTION = 0.1


def case_path(name: str) -> Path:
    """Return the YAML path for a validation case name.

    Parameters
    ----------
    name : str
        One of the keys of CASE_FILES.

    Returns
    -------
    Path
        Absolute path to the configuration file.

    Raises
    ------
    KeyError
        If the case name is not known.
    """
    try:
        return CONFIG_DIR / CASE_FILES[name]
    except KeyError:
        raise KeyError(
            f"unknown validation case {name!r}; known: {sorted(CASE_FILES)}"
        ) from None


def _raw_case(name: str, grid: tuple[int, int] | None) -> dict:
    """The case file as parsed, with the grid overridden when one is given."""
    raw = yaml.safe_load(case_path(name).read_text(encoding="utf-8"))
    if grid is not None:
        raw["domain"]["nx"], raw["domain"]["ny"] = grid
    return raw


def load_case(name: str, grid: tuple[int, int] | None = None) -> SimConfig:
    """Load a validation case, optionally overriding its grid.

    Parameters
    ----------
    name : str
        One of the keys of CASE_FILES.
    grid : tuple[int, int], optional
        (nx, ny) to use instead of the values in the file. Every other
        parameter comes from the file.

    Returns
    -------
    SimConfig
        Validated configuration for the case.
    """
    return SimConfig.from_dict(_raw_case(name, grid))


def load_wall_clustered(name: str, grid: tuple[int, int] | None = None) -> SimConfig:
    """Load a case with y clustered toward both walls as ECR-001 criterion 2 fixes.

    Parameters
    ----------
    name : str
        One of the keys of CASE_FILES.
    grid : tuple[int, int], optional
        (nx, ny) to use instead of the values in the file.

    Returns
    -------
    SimConfig
        The case with ``mesh.y`` replaced by a wall spacing of
        WALL_SPACING_FRACTION * height / ny, the ratio left for the mesh to
        derive. The x axis and every other parameter come from the file.

    Raises
    ------
    ValueError
        If the case file or the grid fails SimConfig's validation, before
        the spacing is computed.
    """
    raw = _raw_case(name, grid)
    # The unclustered case is validated first, so a missing, zero or mistyped
    # height or cell count is refused by SimConfig with its message and not
    # met as a TypeError or a division by zero in the arithmetic below.
    base = SimConfig.from_dict(raw)
    spacing = WALL_SPACING_FRACTION * base.room_height / base.ny
    raw["mesh"] = {**(raw.get("mesh") or {}), "y": {"min_wall_spacing": spacing}}
    return SimConfig.from_dict(raw)


def load_annex20(
    name: str = "annex20", grid: tuple[int, int] | None = None
) -> SimConfig:
    """Load the Annex 20 room with its inlet carrying the specified flux exactly.

    The slot is h = 0.168 m under a 3 m ceiling, so on most grids its edge
    falls inside a cell and the faces the registry gives it span a little
    more or less than h. ADR-012 G (iii) scales the inlet velocity so those
    faces carry u0 h exactly: the ratio of the specified flux, the file's
    velocity times the segment's height, to the flux the covered faces carry
    at the file's velocity, read from the staggered boundary layer on the
    unscaled case so the loader and the solver cannot disagree about a face.
    On a grid where the slot falls on whole faces the factor is one.

    Parameters
    ----------
    name : str
        The case name, "annex20"; a parameter so load_preset can call every
        loader the same way.
    grid : tuple[int, int], optional
        (nx, ny) to use instead of the values in the file.

    Returns
    -------
    SimConfig
        The case with the inlet's ``velocity`` scaled to the grid. The
        inlet's turbulence intensity is unchanged, so its k follows the
        scaled velocity. Every other parameter comes from the file.

    Raises
    ------
    ValueError
        If the case file or the grid fails SimConfig's validation, or the
        boundary layer's: on a grid so coarse that the slot covers no face
        the layer refuses the outlet, which would have no inflow to share,
        before any scaling is attempted.
    """
    raw = _raw_case(name, grid)
    base = SimConfig.from_dict(raw)
    mesh = Mesh(base)
    boundary = StaggeredBoundary(mesh, base)
    inlet = raw["boundaries"][ANNEX20_INLET]
    specified = inlet["velocity"] * (inlet["y_end"] - inlet["y_start"])
    covered = boundary.get_inlet_flux(ANNEX20_INLET)
    inlet["velocity"] = inlet["velocity"] * specified / covered
    return SimConfig.from_dict(raw)


def step_face_index(ny: int) -> int:
    """The index of the y face that load_backstep puts on the step height.

    Parameters
    ----------
    ny : int
        The grid's cell count in y.

    Returns
    -------
    int
        ``round(STEP_FACE_SHARE * ny)``, at least 1 and at most ``ny // 2``,
        the last face the clustering from the floor builds before the mirror.

    Raises
    ------
    ValueError
        If ny is below 6, where no face can be both above the floor cell and
        inside the floor's half of the mesh.
    """
    if ny < 6:
        raise ValueError(f"the step needs at least 6 cells in y, got {ny}")
    return max(1, min(ny // 2, round(STEP_FACE_SHARE * ny)))


def wall_spacing_for_face(length: float, n: int, k: int, target: float) -> float:
    """The wall spacing whose symmetric geometric mesh puts face k at a target.

    Parameters
    ----------
    length : float
        Axis length L.
    n : int
        Number of cells.
    k : int
        Face index, 1 <= k <= n // 2.
    target : float
        Where face k must fall, in (0, k L / n]; above the uniform position
        k L / n the mesh cannot reach it, since clustering toward the wall
        only moves faces toward it.

    Returns
    -------
    float
        The wall-adjacent spacing h in (0, L / n]; L / n when face k's
        uniform position is the target already.

    Raises
    ------
    ValueError
        If k is outside [1, n // 2] or the target outside (0, k L / n].

    Notes
    -----
    face_k(h) rises with h: a larger wall cell and a smaller ratio put every
    face below the midpoint further from the wall. Bisection on h between a
    tiny fraction of L / n and L / n therefore brackets the target. ``Mesh``
    rebuilds the faces from the same two functions, so the face it carries
    is this one to the bisection's precision.
    """
    uniform = length / n
    if not 1 <= k <= n // 2:
        raise ValueError(f"face index must lie in [1, {n // 2}] for {n} cells, got {k}")
    if not 0.0 < target <= k * uniform:
        raise ValueError(
            f"face {k} can reach (0, {k * uniform}] on {n} cells over {length}, "
            f"not {target}"
        )
    if target == k * uniform:
        return uniform

    def face(h: float) -> float:
        ratio = ratio_for_wall_spacing(length, n, h)
        return float(geometric_faces(length, n, ratio)[k])

    lo, hi = 1e-6 * uniform, uniform
    while hi - lo > FACE_TOLERANCE * 1e-3 * uniform:
        mid = 0.5 * (lo + hi)
        if face(mid) > target:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def load_backstep(
    name: str = "backstep", grid: tuple[int, int] | None = None
) -> SimConfig:
    """Load the backward-facing step with the step's corner on grid lines.

    The step is a SOLID block from the inlet wall to x = 0 of the case's
    frame, its top at the step height H (ADR-012 G (ii)). Its corner must
    lie on a face in both axes: the wall stencil puts an obstacle's wall at
    the SOLID cell's face, so a corner off the grid lines moves the step.
    The x axis is uniform and the file's grids put the step's face on one;
    the y axis is clustered toward the floor and the ceiling, and this
    loader derives ``mesh.y.min_wall_spacing`` so that face
    ``step_face_index(ny)`` falls on H on the grid at hand, the same face
    the committed file carries for its own grid.

    Parameters
    ----------
    name : str
        The case name, "backstep"; a parameter so load_preset can call
        every loader the same way.
    grid : tuple[int, int], optional
        (nx, ny) to use instead of the values in the file.

    Returns
    -------
    SimConfig
        The case with ``mesh.y`` replaced by the derived wall spacing. The
        x axis and every other parameter come from the file.

    Raises
    ------
    ValueError
        If the case file or the grid fails SimConfig's validation, if the
        step's face at x = 0 does not fall on an x face of the grid, or if
        the file names no obstacle called BACKSTEP_OBSTACLE.
    """
    raw = _raw_case(name, grid)
    base = SimConfig.from_dict(raw)
    steps = [o for o in base.obstacles if o.name == BACKSTEP_OBSTACLE]
    if len(steps) != 1:
        raise ValueError(
            f"the case needs one obstacle named {BACKSTEP_OBSTACLE!r}, found {len(steps)}"
        )
    step = steps[0]
    dx = base.room_width / base.nx
    cells = step.x_end / dx
    if abs(cells - round(cells)) > FACE_TOLERANCE * base.nx:
        raise ValueError(
            f"the step's face at x = {step.x_end} m is {cells} cells of {dx} m from "
            f"the inlet; the grid must put it on a face"
        )
    spacing = wall_spacing_for_face(
        base.room_height, base.ny, step_face_index(base.ny), step.y_end
    )
    raw["mesh"] = {**(raw.get("mesh") or {}), "y": {"min_wall_spacing": spacing}}
    return SimConfig.from_dict(raw)


def step_corner_offsets(mesh: Mesh, config: SimConfig) -> tuple[float, float]:
    """How far the step's corner lies from the nearest grid line in each axis.

    Parameters
    ----------
    mesh : Mesh
        The grid of a loaded step case.
    config : SimConfig
        The case, with its BACKSTEP_OBSTACLE.

    Returns
    -------
    tuple[float, float]
        The distance from the step's face at x = 0 to the nearest x face,
        and from the step height to the nearest y face, in metres.
    """
    step = next(o for o in config.obstacles if o.name == BACKSTEP_OBSTACLE)
    return (
        float(np.min(np.abs(mesh.x - step.x_end))),
        float(np.min(np.abs(mesh.y - step.y_end))),
    )


def load_preset(case_id: str) -> SimConfig:
    """Load a named grid preset on the mesh its name promises.

    Parameters
    ----------
    case_id : str
        One of the keys of CASE_GRIDS.

    Returns
    -------
    SimConfig
        The preset's case at its grid, by load_wall_clustered for a preset
        in WALL_CLUSTERED_GRIDS, by load_annex20 for one in ANNEX20_GRIDS,
        by load_backstep for one in BACKSTEP_GRIDS and by load_case for
        every other.

    Raises
    ------
    KeyError
        If the preset is not known.
    """
    try:
        kind, nx, ny = CASE_GRIDS[case_id]
    except KeyError:
        raise KeyError(
            f"unknown grid preset {case_id!r}; known: {sorted(CASE_GRIDS)}"
        ) from None
    if case_id in WALL_CLUSTERED_GRIDS:
        loader = load_wall_clustered
    elif case_id in ANNEX20_GRIDS:
        loader = load_annex20
    elif case_id in BACKSTEP_GRIDS:
        loader = load_backstep
    else:
        loader = load_case
    return loader(kind, grid=(nx, ny))


def with_velocity_step(config: SimConfig) -> SimConfig:
    """A copy of a case configuration that stops by velocity_step.

    A staggered solve that must reproduce a field saved under velocity_step
    is pinned to that rule this way (scripts/stopping_probe.py,
    scripts/self_convergence.py). Nothing else changes.

    Parameters
    ----------
    config : SimConfig
        A loaded case configuration. Not modified.

    Returns
    -------
    SimConfig
        A shallow copy with ``stopping_rule`` set to velocity_step. The
        error_estimate tolerances stay as loaded, unused.
    """
    out = copy.copy(config)
    out.stopping_rule = VELOCITY_STEP
    return out


def with_pressure_preconditioner(config: SimConfig, preconditioner: str) -> SimConfig:
    """A copy of a case configuration that names its pressure preconditioner.

    The harness runs a case under the preconditioner its method label names
    (scripts/benchmark.py), whatever the case file says. Nothing else changes.

    Parameters
    ----------
    config : SimConfig
        A loaded case configuration. Not modified.
    preconditioner : str
        One of ``PRESSURE_PRECONDITIONERS``.

    Returns
    -------
    SimConfig
        A shallow copy with ``pressure_preconditioner`` set.

    Raises
    ------
    ValueError
        If the preconditioner is not one of ``PRESSURE_PRECONDITIONERS``.
    """
    if preconditioner not in PRESSURE_PRECONDITIONERS:
        raise ValueError(
            f"pressure preconditioner must be one of {list(PRESSURE_PRECONDITIONERS)}, "
            f"got {preconditioner!r}"
        )
    out = copy.copy(config)
    out.pressure_preconditioner = preconditioner
    return out
