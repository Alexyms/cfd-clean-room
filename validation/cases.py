"""Load validation case configurations from the committed YAML files.

Each validation case has exactly one configuration, checked in under
configs/, so the tests and the benchmark harness run the same solve from
the same parameters. The only override this module accepts is the grid,
which is the one axis a refinement study varies.
"""

from pathlib import Path

import yaml

from src.config import SimConfig

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"

CASE_FILES: dict[str, str] = {
    "poiseuille": "validation_poiseuille.yaml",
    "cavity": "validation_cavity.yaml",
}

# Named grid presets: identifier -> (case, nx, ny). The harness and the field
# viewer both accept these identifiers so a benchmark row and a picture of the
# same solve are named the same way.
CASE_GRIDS: dict[str, tuple[str, int, int]] = {
    "val001_80x40": ("poiseuille", 80, 40),
    "val002_20x20": ("cavity", 20, 20),
    "val002_40x40": ("cavity", 40, 40),
    "val002_80x80": ("cavity", 80, 80),
}


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
    path = case_path(name)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if grid is not None:
        raw["domain"]["nx"], raw["domain"]["ny"] = grid
    return SimConfig.from_dict(raw)
