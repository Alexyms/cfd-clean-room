"""Load validation case configurations from the committed YAML files.

Each validation case has exactly one configuration, checked in under
configs/, so the tests and the benchmark harness run the same solve from
the same parameters. The overrides this module makes are the grid, which is
the one axis a refinement study varies; the wall clustering ECR-001
acceptance criterion 2 fixes, which load_wall_clustered derives from the
case's own height and cell count rather than from a second file; and the
stopping rule, which with_velocity_step sets to the one the collocated
solver accepts. A named preset is loaded by load_preset, which picks the
loader, so no consumer repeats that choice.
"""

import copy
from pathlib import Path

import yaml

from src.config import VELOCITY_STEP, SimConfig

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"

CASE_FILES: dict[str, str] = {
    "poiseuille": "validation_poiseuille.yaml",
    "cavity": "validation_cavity.yaml",
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
}

# Presets loaded by load_wall_clustered; every other preset takes its whole
# mesh from the case file.
WALL_CLUSTERED_GRIDS: frozenset[str] = frozenset({"val001_80x40_stretched"})

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
    """
    raw = _raw_case(name, grid)
    spacing = WALL_SPACING_FRACTION * raw["domain"]["height"] / raw["domain"]["ny"]
    raw["mesh"] = {**(raw.get("mesh") or {}), "y": {"min_wall_spacing": spacing}}
    return SimConfig.from_dict(raw)


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
        in WALL_CLUSTERED_GRIDS and by load_case for every other.

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
    loader = load_wall_clustered if case_id in WALL_CLUSTERED_GRIDS else load_case
    return loader(kind, grid=(nx, ny))


def with_velocity_step(config: SimConfig) -> SimConfig:
    """A copy of a case configuration that stops by velocity_step, for the collocated solver.

    The collocated solver refuses error_estimate: its walls leak mass, so the
    continuity condition could never hold. It keeps the rule every collocated
    result was produced under until it is retired, and nothing else changes.
    A staggered solve that must reproduce a field saved under velocity_step
    is pinned the same way.

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
