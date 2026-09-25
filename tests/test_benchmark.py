"""Unit tests for the benchmark harness summary printer in scripts/benchmark.py."""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import benchmark  # noqa: E402 -- scripts/ is not a package; path set above

from src.boundary import BoundaryManager  # noqa: E402 -- follows sys.path.insert
from src.config import SimConfig  # noqa: E402 -- follows sys.path.insert
from src.mesh import Mesh  # noqa: E402 -- follows sys.path.insert
from src.solver_ns import (  # noqa: E402 -- follows sys.path.insert
    IterationState,
    NavierStokesSolver,
)
from src.solver_staggered import (  # noqa: E402 -- follows sys.path.insert
    StaggeredSolver,
)
from validation.cases import (  # noqa: E402 -- follows sys.path.insert
    case_path,
    load_case,
)
from validation.metrics import (  # noqa: E402 -- follows sys.path.insert
    cavity_true_centerline_errors,
)


def _record(case: str, procs: int, wall: float, outer: int = 100) -> dict:
    """A minimal record with the fields the summary reads."""
    return {
        "method": "collocated-jacobi",
        "case": case,
        "environment": {"concurrent_processes": procs},
        "outcome": {"converged": True},
        "accuracy": {"value": 0.1},
        "work": {"outer_iterations": outer, "cell_updates": 1000},
        "time": {"wall_seconds": wall},
    }


def _write(path: Path, records: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return path


@pytest.mark.unit
def test_summary_separates_rows_by_concurrent_processes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Rows of one case taken under different loads are never pooled into one wall-time row."""
    path = _write(
        tmp_path / "results.jsonl",
        [
            _record("val002_40x40", 1, 84.0),
            _record("val002_40x40", 1, 86.0),
            _record("val002_40x40", 2, 660.0),
        ],
    )
    benchmark.print_summary(path)
    lines = capsys.readouterr().out.splitlines()
    rows = [line for line in lines if line.startswith("collocated-jacobi")]
    assert len(rows) == 2
    procs_one = next(line for line in rows if " 1  2 " in line)
    procs_two = next(line for line in rows if " 2  1 " in line)
    assert "660.0" not in procs_one
    assert "660.0" in procs_two


@pytest.mark.unit
def test_summary_flags_case_recorded_under_mixed_load(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A case with rows at more than one concurrency gets a note naming the loads."""
    path = _write(
        tmp_path / "results.jsonl",
        [_record("val002_80x80", 1, 600.0), _record("val002_80x80", 2, 669.0)],
    )
    benchmark.print_summary(path)
    out = capsys.readouterr().out
    assert (
        "note: collocated-jacobi val002_80x80 has rows at concurrent_processes [1, 2]"
        in out
    )
    assert "not comparable" in out


@pytest.mark.unit
def test_summary_notes_mixed_load_across_cases(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Different cases at different loads get a table-level note, as the 80x80 row did."""
    path = _write(
        tmp_path / "results.jsonl",
        [_record("val002_40x40", 1, 85.0), _record("val002_80x80", 2, 669.0)],
    )
    benchmark.print_summary(path)
    out = capsys.readouterr().out
    assert "note: rows above were recorded at concurrent_processes [1, 2]" in out
    assert "note: collocated-jacobi val002" not in out


@pytest.mark.unit
def test_summary_has_no_note_when_load_is_uniform(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Uniform load produces the table and nothing else."""
    path = _write(
        tmp_path / "results.jsonl",
        [_record("val001_80x40", 1, 150.0), _record("val001_80x40", 1, 152.0)],
    )
    benchmark.print_summary(path)
    out = capsys.readouterr().out
    assert "note:" not in out
    assert "procs" in out.splitlines()[0]


@pytest.mark.unit
def test_mixed_load_cases_detects_only_multi_load_pairs() -> None:
    """The helper reports exactly the (method, case) pairs with more than one load."""
    groups = {
        ("m", "a", 1): [],
        ("m", "a", 2): [],
        ("m", "b", 1): [],
    }
    assert benchmark.mixed_load_cases(groups) == {("m", "a")}


@pytest.mark.unit
def test_concurrent_rejects_values_below_one(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """--concurrent counts this process too, so zero and negatives are refused at parse time."""
    for value in ("0", "-1"):
        with pytest.raises(SystemExit) as exc:
            benchmark.main(["--concurrent", value, "--summary"])
        assert exc.value.code == 2
        err = capsys.readouterr().err
        assert "--concurrent" in err
        assert f"must be at least 1, got {value}" in err


@pytest.mark.unit
def test_summary_never_pools_two_references(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """One case scored against the old and the corrected Ghia table gets two error ranges.

    Without the reference in the row key the stored ghia_1982_re100 rows and
    the ghia_1982_re100_r2 rows would print as one range spanning both tables.
    """
    old = _record("val002_40x40", 1, 85.0)
    old["accuracy"] = {"value": 0.1893, "reference": "ghia_1982_re100"}
    new = _record("val002_40x40", 1, 86.0)
    new["accuracy"] = {"value": 0.0511, "reference": "ghia_1982_re100_r2"}
    benchmark.print_summary(_write(tmp_path / "results.jsonl", [old, new]))
    out = capsys.readouterr().out
    rows = [line for line in out.splitlines() if line.startswith("collocated-jacobi")]
    assert len(rows) == 2
    old_row = next(line for line in rows if line.endswith(" ghia_1982_re100"))
    new_row = next(line for line in rows if line.endswith(" ghia_1982_re100_r2"))
    assert "1.893e-01" in old_row
    assert "5.110e-02" not in old_row
    assert "5.110e-02" in new_row
    assert "1.893e-01" not in new_row
    assert (
        "note: collocated-jacobi val002_40x40 has rows scored against "
        "['ghia_1982_re100', 'ghia_1982_re100_r2']"
    ) in out


@pytest.mark.unit
def test_summary_never_pools_two_metrics(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """One case scored by the offset and the true-centerline metric gets two error ranges.

    Both carry the same reference, so only the metric in the row key keeps
    them apart.
    """
    old = _record("val002_40x40", 1, 85.0)
    old["accuracy"] = {
        "metric": "max_normalized_centerline_error",
        "value": 0.0224,
        "reference": "ghia_1982_re100_r2",
    }
    new = _record("val002_40x40", 1, 86.0)
    new["accuracy"] = {
        "metric": "max_normalized_centerline_error_r2",
        "value": 0.0080,
        "reference": "ghia_1982_re100_r2",
    }
    benchmark.print_summary(_write(tmp_path / "results.jsonl", [old, new]))
    out = capsys.readouterr().out
    rows = [line for line in out.splitlines() if line.startswith("collocated-jacobi")]
    assert len(rows) == 2
    old_row = next(line for line in rows if " max_normalized_centerline_error " in line)
    new_row = next(
        line for line in rows if " max_normalized_centerline_error_r2 " in line
    )
    assert "2.240e-02" in old_row
    assert "8.000e-03" not in old_row
    assert "8.000e-03" in new_row
    assert "2.240e-02" not in new_row
    assert (
        "note: collocated-jacobi val002_40x40 has rows measured by "
        "['max_normalized_centerline_error', 'max_normalized_centerline_error_r2']"
    ) in out
    assert "scored against" not in out


@pytest.mark.unit
def test_harness_scores_the_cavity_on_the_true_centerlines() -> None:
    """A cavity record's accuracy is the true-centerline metric, by name and value."""
    config = load_case("cavity", grid=(16, 16))
    mesh = Mesh(config)
    x, y = np.meshgrid(np.asarray(mesh.xc), np.asarray(mesh.yc))
    u, v = 0.3 + 0.8 * x, 0.3 + 0.8 * y
    accuracy = benchmark.accuracy_of("cavity", config, mesh, u, v)
    assert accuracy["metric"] == "max_normalized_centerline_error_r2"
    assert accuracy == cavity_true_centerline_errors(config, mesh, u, v).as_dict()


@pytest.mark.integration
def test_harness_row_takes_the_cap_from_the_solver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A capped error_estimate solve is recorded as not converged, with its rule.

    Its last residual is below convergence_tol 10, so the old computation from
    the residual would have recorded it as converged.
    """
    raw = yaml.safe_load(case_path("cavity").read_text(encoding="utf-8"))
    raw["domain"]["nx"] = raw["domain"]["ny"] = 6
    raw["solver"].update(
        stopping_rule="error_estimate", convergence_tol=10.0, max_simple_iter=20
    )
    config = SimConfig.from_dict(raw)
    monkeypatch.setitem(benchmark.CASES, "tiny_cavity", ("cavity", 6, 6))
    monkeypatch.setattr(benchmark, "load_case", lambda kind, grid: config)
    row = benchmark.run_case("tiny_cavity", "staggered-jacobi", 10, 1)
    assert row["outcome"] == {"converged": False, "stop_reason": "max_simple_iter"}
    assert row["work"]["outer_iterations"] == 20
    assert row["trajectory"][-1]["residual"] < 10.0
    assert row["params"]["stopping_rule"] == "error_estimate"


@pytest.mark.unit
def test_summary_never_pools_two_rules(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Rows that differ only in stopping rule get two rows; a row without the key ran velocity_step."""
    old = _record("val001_80x40", 1, 85.0, outer=570)
    new = _record("val001_80x40", 1, 96.0, outer=3154)
    new["params"] = {"stopping_rule": "error_estimate"}
    benchmark.print_summary(_write(tmp_path / "results.jsonl", [old, new]))
    out = capsys.readouterr().out
    rows = [line for line in out.splitlines() if line.startswith("collocated-jacobi")]
    assert len(rows) == 2
    old_row = next(line for line in rows if " velocity_step " in line)
    new_row = next(line for line in rows if " error_estimate " in line)
    assert " 570..570 " in old_row and "3154" not in old_row
    assert " 3154..3154" in new_row and "570" not in new_row
    assert (
        "note: collocated-jacobi val001_80x40 has rows under "
        "['error_estimate', 'velocity_step']"
    ) in out


@pytest.mark.integration
def test_collocated_channel_row_is_the_pre_branch_solve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The collocated channel runs velocity_step: its params and fields are the case's as
    it was before it named a rule, and the stop carries the one velocity-step label."""
    fields: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    class Spy(NavierStokesSolver):
        def solve_steady(
            self, on_iteration: Callable[[IterationState], None] | None = None
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            fields.append(super().solve_steady(on_iteration))
            return fields[-1]

    monkeypatch.setattr(benchmark, "NavierStokesSolver", Spy)
    monkeypatch.setitem(benchmark.CASES, "tiny_channel", ("poiseuille", 12, 6))
    row = benchmark.run_case("tiny_channel", "collocated-jacobi", 10, 1)
    raw = yaml.safe_load(case_path("poiseuille").read_text(encoding="utf-8"))
    raw["domain"]["nx"], raw["domain"]["ny"] = 12, 6
    for key in ("stopping_rule", "iteration_error_tol", "mass_imbalance_tol"):
        raw["solver"].pop(key, None)
    before = SimConfig.from_dict(raw)
    mesh = Mesh(before)
    expected = NavierStokesSolver(mesh, before, BoundaryManager(mesh, before))
    for got, want in zip(fields[0], expected.solve_steady(), strict=True):
        assert np.array_equal(got, want)
    assert row["params"] == benchmark.solver_parameters(before)
    assert row["params"]["stopping_rule"] == "velocity_step"
    assert row["outcome"] == {"converged": True, "stop_reason": "residual_below_tol"}


@pytest.mark.integration
def test_staggered_velocity_step_stop_has_the_collocated_label(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The cavity still runs velocity_step; its staggered row says what every stored row says."""
    monkeypatch.setitem(benchmark.CASES, "tiny_cavity", ("cavity", 6, 6))
    row = benchmark.run_case("tiny_cavity", "staggered-jacobi", 10, 1)
    assert row["params"]["stopping_rule"] == "velocity_step"
    assert row["outcome"] == {"converged": True, "stop_reason": "residual_below_tol"}


@pytest.mark.unit
def test_harness_builds_a_wall_clustered_preset_on_its_clustered_mesh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The preset's mesh reaches the solver; the solve is stopped at construction."""
    meshes: list[Mesh] = []

    class ConstructedError(Exception):
        pass

    class Spy(StaggeredSolver):
        def __init__(self, mesh: Mesh, *args: object) -> None:
            meshes.append(mesh)
            raise ConstructedError

    monkeypatch.setattr(benchmark, "StaggeredSolver", Spy)
    monkeypatch.setitem(benchmark.CASES, "tiny_clustered", ("poiseuille", 12, 6))
    monkeypatch.setattr(benchmark, "WALL_CLUSTERED_GRIDS", {"tiny_clustered"})
    with pytest.raises(ConstructedError):
        benchmark.run_case("tiny_clustered", "staggered-jacobi", 10, 1)
    assert meshes[0].dy_cell[0] == pytest.approx(0.1 * 0.5 / 6, rel=1e-12)
    assert meshes[0].stretch_ratio_x == 1.0
