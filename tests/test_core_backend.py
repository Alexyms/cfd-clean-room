"""The backend switch end to end (prompt 96, ADR-017 E): a scenario run under ``core``.

``solver.backend: core`` has ``time_integration.run`` build ``core_backend.CoreRun``
and every steady solve a ``CoreSolver``. Here a short dust release on 80x30 runs
from rest under each backend: the core's run against the reference's by ADR-017
H's comparator (``scripts/compare_runs.py``); its restart from a checkpoint in the
march against the uninterrupted core run, bit for bit; the monitor on the core's
readings against the monitor on the fields, bit for bit; and the refusals of a
restart on the other backend. A sensor is added at the release's cell, so that
alerts open and close in the run's twelve seconds.

Skipped with the other parity modules where ``_cfdcore`` cannot be imported (the
Windows suite), and an error instead with ``CFD_CORE_REQUIRED=1``.
"""

import copy
import os
import sys
from pathlib import Path

import pytest
import yaml

import src.time_integration as time_integration
from src import core_backend
from src.boundary_staggered import StaggeredBoundary
from src.config import SimConfig
from src.io_manager import RunFile
from src.mesh import Mesh
from src.time_integration import Provenance, run
from validation.cases import with_backend
from validation.transport_cases import conservation_keys

if os.environ.get("CFD_CORE_REQUIRED") == "1":
    import _cfdcore  # noqa: F401 -- the import is the requirement
else:
    pytest.importorskip(
        "_cfdcore",
        reason=(
            "the compiled core is not built: "
            "pip install --no-build-isolation ./core under WSL2 (ADR-017 A)"
        ),
    )

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import compare_runs  # noqa: E402 -- scripts/ is not a package; path set above

pytestmark = [pytest.mark.parity, pytest.mark.integration]

CONFIGS = ROOT / "configs"
PROVENANCE = Provenance("test", "2026-11-26T00:00:00", "test-machine", 0)
# The march's checkpoint cadence, and the step after which the run is stopped.
CADENCE, STOP_AFTER = 200, 450


def _raw(backend: str) -> dict:
    """The short dust release on 80x30 on BACKEND, a sensor at the release's cell."""
    raw = yaml.safe_load(
        (CONFIGS / "scenario_equipment_dust.yaml").read_text(encoding="utf-8")
    )
    base = yaml.safe_load(
        (CONFIGS / "clean_room_default.yaml").read_text(encoding="utf-8")
    )
    raw["scenario"]["base"] = str(CONFIGS / "clean_room_default.yaml")
    raw["domain"] = {"nx": 80, "ny": 30}
    raw["solver"]["mass_imbalance_tol"] = (
        conservation_keys()["conservation_limit"] * 1.2 * (0.1 * 0.1) / 60.0
    )
    raw["solver"]["t_end"] = 12.0
    raw["solver"]["checkpoint_interval"] = CADENCE
    raw["solver"]["backend"] = backend
    raw["scenario"]["events"][0]["start"] = 2.0
    raw["scenario"]["events"][0]["duration"] = 4.0
    raw["sensors"] = [*base["sensors"], {"name": "at_release", "x": 3.85, "y": 2.05}]
    return raw


class _StopError(RuntimeError):
    """The crash a restart continues from."""


@pytest.fixture(scope="module")
def runs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """The reference's run, the core's, the core's with the monitor on the fields,
    and the core's stopped after STOP_AFTER steps and restarted."""
    out = tmp_path_factory.mktemp("backend")
    paths = {name: out / f"{name}.h5" for name in ("ref", "core", "fields", "cut")}
    run(SimConfig.from_dict(_raw("reference")), paths["ref"], provenance=PROVENANCE)
    core = SimConfig.from_dict(_raw("core"))
    run(core, paths["core"], provenance=PROVENANCE)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(core_backend.CoreRun, "monitor_reads_fields", True)
        run(core, paths["fields"], provenance=PROVENANCE)

    def stop(step: object) -> None:
        if step.step == STOP_AFTER:
            raise _StopError

    with pytest.raises(_StopError):
        run(core, paths["cut"], provenance=PROVENANCE, on_step=stop)
    restarted = run(core, restart=paths["cut"])
    assert restarted.restarted_from == CADENCE * (STOP_AFTER // CADENCE)
    return paths


def test_the_key_builds_the_cores_run_and_solver(tmp_path: Path) -> None:
    """Under ``core``: ``CoreRun`` and ``CoreSolver``, whatever reference is named.

    Defect caught: the factory building the reference under ``core``.
    """
    config = SimConfig.from_dict(_raw("core"))
    state = time_integration._run_state(config, RunFile(tmp_path / "c.h5"), None)
    assert type(state) is core_backend.CoreRun
    assert not core_backend.CoreRun.monitor_reads_fields
    mesh = Mesh(config)
    boundary = StaggeredBoundary(mesh, config)
    solver = core_backend.steady_solver(mesh, config, boundary)
    assert type(solver) is core_backend.CoreSolver
    named = core_backend.steady_solver(
        mesh, config, boundary, reference=time_integration.StaggeredSolver
    )
    assert type(named) is core_backend.CoreSolver
    with pytest.raises(NotImplementedError, match="ADR-017 F"):
        solver.solve_transient()


def test_the_core_run_makes_the_references_decisions(runs: dict[str, Path]) -> None:
    """The core's run against the reference's: ADR-017 H's PASS, stops equal.

    Alerts open at the release's sensor, so the alert items compare something.
    """
    comparison = compare_runs.compare(runs["ref"], runs["core"])
    assert comparison.verdict == "PASS", compare_runs.report(comparison)
    stops = [item for item in comparison.items if item.criterion == "stop"]
    assert stops and all(item.detail["apart"] == 0 for item in stops)
    alerts = RunFile(runs["ref"]).read_alerts()
    assert len(alerts.step_on) > 0
    assert RunFile(runs["core"]).read_config().mapping["solver"]["backend"] == "core"


def test_a_restart_under_core_is_the_uninterrupted_run_bit_for_bit(
    runs: dict[str, Path],
) -> None:
    """Stopped after a step in the march, restarted from its checkpoint: bitwise."""
    comparison = compare_runs.compare(runs["core"], runs["cut"])
    assert all(comparison.bitwise.values()), comparison.bitwise
    assert comparison.verdict == "PASS"


def test_the_monitor_on_the_readings_is_the_monitor_on_the_fields(
    runs: dict[str, Path],
) -> None:
    """The two updates on the same run: alerts, latencies and sensor table bitwise.

    Defects caught: the readings taken a step late, or a channel from another
    class, which move an alert or a reading.
    """
    comparison = compare_runs.compare(runs["fields"], runs["core"])
    for name in ("alerts", "latencies", "sensors"):
        assert comparison.bitwise[name], name
    assert all(comparison.bitwise.values()), comparison.bitwise


def test_a_run_file_of_one_backend_is_refused_for_a_restart_on_the_other(
    runs: dict[str, Path], tmp_path: Path
) -> None:
    """A core run restarted under the reference, and a reference run under the core."""
    for source, backend in (("core", "reference"), ("ref", "core")):
        cut = tmp_path / f"{source}.h5"
        cut.write_bytes(runs[source].read_bytes())
        checkpoint = runs[source].with_name(runs[source].stem + ".checkpoint.h5")
        if checkpoint.exists():
            cut.with_name(cut.stem + ".checkpoint.h5").write_bytes(
                checkpoint.read_bytes()
            )
        raw = copy.deepcopy(_raw(backend))
        raw["solver"]["t_end"] = 14.0
        with pytest.raises(ValueError, match=r"solver\.backend differs"):
            run(SimConfig.from_dict(raw), restart=cut)


def test_a_with_backend_copy_runs_on_the_core_and_its_file_names_it(
    tmp_path: Path,
) -> None:
    """A case copied by ``with_backend`` runs on the core and records that it did.

    The march and every flow solve run on the core, the run file's stored
    configuration names the backend, and a restart on the reference, whose
    configuration does not, is refused (test 96 S3).

    Defect caught: the key set on the copy's attribute and not in its mapping,
    which ran on the core but wrote a file that a restart on the reference
    accepted.
    """
    raw = _raw("reference")
    del raw["solver"]["backend"]
    reference = SimConfig.from_dict(raw)
    core = with_backend(reference, "core")
    seen: list[str] = []
    with pytest.MonkeyPatch.context() as patch:
        for owner, name in (
            (core_backend.CoreRun, "march"),
            (core_backend.CoreSolver, "solve_steady"),
        ):
            inner = getattr(owner, name)

            def spy(self, *args, _inner=inner, _tag=f"{owner.__name__}.{name}", **kw):
                seen.append(_tag)
                return _inner(self, *args, **kw)

            patch.setattr(owner, name, spy)
        path = tmp_path / "named.h5"
        run(core, path, provenance=PROVENANCE)
    assert "CoreRun.march" in seen and "CoreSolver.solve_steady" in seen
    assert RunFile(path).read_config().mapping["solver"]["backend"] == "core"
    longer = copy.deepcopy(raw)
    longer["solver"]["t_end"] = 14.0
    with pytest.raises(ValueError, match=r"solver\.backend differs"):
        run(SimConfig.from_dict(longer), restart=path)
