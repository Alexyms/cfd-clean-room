"""The step 0 probe's field path, read from the committed report, for comparison.

ECR-002 step 4 built ADR-012 D's viscosity field into src/momentum.py as the
probe of step 0 wrote it, ``frozen34.py``'s ``FrozenPredictor``. The probe
file itself lives under results/ and is not committed; its text is appendix B
of docs/reports/ecr002_step0_frozen_viscosity.md. This module takes the two
means and the class from that appendix by AST and executes them against the
committed momentum module, so a test compares the built code with the probe
as published rather than with a copy that could drift. Nothing else of the
probe (its runner, its outlet solver) is executed.
"""

import ast
import re
from pathlib import Path

import numpy as np

from src import momentum

REPORT = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "reports"
    / "ecr002_step0_frozen_viscosity.md"
)
APPENDIX = "## Appendix B: frozen34.py"
TAKEN = ("lerp", "harmonic", "FrozenPredictor")


def probe_source() -> str:
    """The text of frozen34.py as appendix B prints it."""
    text = REPORT.read_text(encoding="utf-8")
    start = text.index(APPENDIX)
    block = re.search(r"```python\n(.*?)\n```", text[start:], re.S)
    if block is None:
        raise ValueError(f"no python block under '{APPENDIX}' in {REPORT}")
    return block.group(1)


def load_frozen_predictor() -> type:
    """The probe's FrozenPredictor, subclassing the committed MomentumPredictor.

    Returns
    -------
    type
        The class, constructed as ``FrozenPredictor(mesh, config, boundary,
        mu_t, deferred=True, sweeps=1, stress=True, form="b")`` with
        ``mu_t`` the dynamic eddy viscosity per cell; its ``predict(u, v,
        p)`` runs the field path with ``mu + mu_t``.
    """
    source = probe_source()
    tree = ast.parse(source)
    pieces = [
        ast.get_source_segment(source, node)
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in TAKEN
    ]
    if len(pieces) != len(TAKEN):
        raise ValueError(f"appendix B lacks one of {TAKEN}")
    namespace = {
        "np": np,
        "MomentumCoefficients": momentum.MomentumCoefficients,
        "MomentumPrediction": momentum.MomentumPrediction,
        "MomentumPredictor": momentum.MomentumPredictor,
        "_Orientation": momentum._Orientation,
    }
    # The executed text is the committed report's, the probe as reviewed.
    exec(compile("\n\n".join(pieces), str(REPORT), "exec"), namespace)
    return namespace["FrozenPredictor"]
