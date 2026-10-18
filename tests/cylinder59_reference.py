"""The step 6 probe's implicit Euler step, read from the committed probe, for comparison.

ADR-014 decision 3 builds the time derivative into src/momentum.py as the
probe of prompt 59 wrote it for one implicit Euler step, the
``ImplicitEulerPredictor`` of docs/reports/probe59/cylinder59.py, which drove
the steady loop one step at a time by replacing its fields at rest with the
previous step's faces (``fields_of``). This module takes that class and that
function from the committed probe by AST and executes them against the
committed modules, so a test compares the built code with the probe as
committed rather than with a copy that could drift, as
tests/frozen34_reference.py does for the step 0 probe. Nothing else of the
probe (its runner, its configuration) is executed.
"""

import ast
from collections.abc import Callable
from pathlib import Path

import numpy as np

from src.boundary_staggered import StaggeredBoundary
from src.config import SimConfig
from src.mesh import Mesh
from src.momentum import MomentumCoefficients, MomentumPredictor, _Orientation

PROBE = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "reports"
    / "probe59"
    / "cylinder59.py"
)
TAKEN = ("ImplicitEulerPredictor", "fields_of")


def _load() -> dict[str, object]:
    """The probe's two definitions, executed in a namespace of committed names."""
    source = PROBE.read_text(encoding="utf-8")
    pieces = [
        ast.get_source_segment(source, node)
        for node in ast.parse(source).body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in TAKEN
    ]
    if len(pieces) != len(TAKEN):
        raise ValueError(f"{PROBE} lacks one of {TAKEN}")
    namespace: dict[str, object] = {
        "np": np,
        "Callable": Callable,
        "Mesh": Mesh,
        "SimConfig": SimConfig,
        "StaggeredBoundary": StaggeredBoundary,
        "MomentumPredictor": MomentumPredictor,
        "MomentumCoefficients": MomentumCoefficients,
        "_Orientation": _Orientation,
    }
    exec("\n\n".join(piece for piece in pieces if piece is not None), namespace)
    return namespace


def load_implicit_euler_predictor() -> type[MomentumPredictor]:
    """The probe's ImplicitEulerPredictor, subclassing the committed MomentumPredictor.

    Returns
    -------
    type
        The class, constructed as ``ImplicitEulerPredictor(mesh, config,
        boundary, dt)``; set ``u_old`` and ``v_old`` to the previous level's
        faces before ``predict(u, v, p)``.
    """
    return _load()["ImplicitEulerPredictor"]  # type: ignore[return-value]


def load_fields_of() -> Callable[
    [np.ndarray, np.ndarray, np.ndarray],
    Callable[[Mesh], tuple[np.ndarray, np.ndarray, np.ndarray]],
]:
    """The probe's ``fields_of``: a stand-in for allocate_fields handing out a kept state."""
    return _load()["fields_of"]  # type: ignore[return-value]
