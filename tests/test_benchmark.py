"""Unit tests for the benchmark harness summary printer in scripts/benchmark.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import benchmark  # noqa: E402 -- scripts/ is not a package; path set above


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
