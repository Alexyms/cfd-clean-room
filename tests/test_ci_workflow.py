"""CI's core job runs every parity module (prompt 96).

The parity modules skip where ``_cfdcore`` cannot be imported, so the main test
job passes without them; only the core job runs them, and it names each one. A
module marked ``parity`` that the job does not name would never run in CI.
"""

import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"


def _parity_modules() -> set[str]:
    """Every test module that marks its tests ``parity``."""
    return {
        f"tests/{path.name}"
        for path in (ROOT / "tests").glob("test_*.py")
        if re.search(r"pytest\.mark\.parity\b", path.read_text(encoding="utf-8"))
    }


def _core_job_modules() -> set[str]:
    """The test modules the core job's parity step names."""
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    steps = workflow["jobs"]["core"]["steps"]
    (step,) = [s for s in steps if s.get("name") == "Run the parity tests"]
    return set(re.findall(r"tests/test_\w+\.py", step["run"]))


@pytest.mark.unit
def test_the_core_job_names_every_parity_module() -> None:
    """The two sets are equal, and the backend's module is among them."""
    named = _core_job_modules()
    assert named == _parity_modules()
    assert "tests/test_core_backend.py" in named
