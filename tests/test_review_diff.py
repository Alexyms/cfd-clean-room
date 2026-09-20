"""Unit tests for the review diff preparation in .github/scripts/review_diff.py.

The module is imported by path because .github/scripts is not a package and
review.py itself needs API clients that the test environment does not have.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / ".github" / "scripts"))

import review_diff  # noqa: E402 -- .github/scripts is not a package; path set above


def _block(path: str, added: int, removed: int, pad: int = 0) -> str:
    """One file's unified diff block with the given line counts."""
    lines = [
        f"diff --git a/{path} b/{path}",
        "index 0000000..1111111 100644",
        f"--- a/{path}",
        f"+++ b/{path}",
        "@@ -1,1 +1,1 @@",
    ]
    lines += [f"-old {i}" for i in range(removed)]
    lines += [f"+new {i}" for i in range(added)]
    lines += [" context" + "x" * pad]
    return "\n".join(lines) + "\n"


@pytest.mark.unit
def test_split_counts_hunk_lines_and_skips_headers() -> None:
    """Added and removed counts come from hunks; --- and +++ headers are not counted."""
    blocks = review_diff.split_diff(_block("a.py", 3, 2) + _block("b.py", 0, 1))
    assert [(b.path, b.added, b.removed) for b in blocks] == [
        ("a.py", 3, 2),
        ("b.py", 0, 1),
    ]


@pytest.mark.unit
def test_removed_line_starting_with_dashes_is_not_a_header() -> None:
    """A removed content line beginning with -- is counted as a removal."""
    diff = "\n".join(
        [
            "diff --git a/x.md b/x.md",
            "--- a/x.md",
            "+++ b/x.md",
            "@@ -1 +1 @@",
            "--- a heading-ish removed line",
            "+new",
        ]
    )
    (block,) = review_diff.split_diff(diff)
    assert (block.added, block.removed) == (1, 1)


@pytest.mark.unit
def test_excluded_file_is_removed_and_summarised() -> None:
    """The jsonl block is not in the text; its line counts are in the notes."""
    diff = _block("benchmarks/results.jsonl", 9, 0) + _block(
        "scripts/benchmark.py", 5, 1
    )
    prepared = review_diff.prepare_diff(diff)
    assert [b.path for b in prepared.excluded] == ["benchmarks/results.jsonl"]
    assert "benchmarks/results.jsonl" not in prepared.text
    assert prepared.text == _block("scripts/benchmark.py", 5, 1)
    (note,) = prepared.notes()
    assert "benchmarks/results.jsonl" in note
    assert "+9 lines, -0 lines" in note
    assert not prepared.truncated


@pytest.mark.unit
def test_other_files_pass_through_byte_for_byte() -> None:
    """With the exclusion applied, every other file is exactly its original block."""
    blocks = [
        _block("benchmarks/results.jsonl", 1, 0),
        _block("src/config.py", 2, 2),
        _block("validation/cases.py", 4, 0),
        _block("tests/test_config.py", 1, 0),
    ]
    prepared = review_diff.prepare_diff("".join(blocks))
    assert prepared.text == "".join(blocks[1:])


@pytest.mark.unit
def test_truncation_names_dropped_paths_and_keeps_later_files_that_fit() -> None:
    """A file over the remaining budget is dropped whole and named; smaller later files still arrive."""
    big = _block("scripts/gen_system_map.py", 1, 0, pad=500)
    small_a = _block("src/solver_ns.py", 1, 0)
    small_b = _block("tests/test_solver_ns.py", 1, 0)
    prepared = review_diff.prepare_diff(
        small_a + big + small_b, limit=len(small_a) + len(small_b)
    )
    assert [b.path for b in prepared.kept] == [
        "src/solver_ns.py",
        "tests/test_solver_ns.py",
    ]
    assert [b.path for b in prepared.dropped] == ["scripts/gen_system_map.py"]
    assert prepared.truncated
    (note,) = prepared.notes()
    assert "scripts/gen_system_map.py" in note
    assert "DROPPED IN FULL" in note


@pytest.mark.unit
def test_no_truncation_note_when_everything_fits() -> None:
    """Under the limit, nothing is dropped and no note is emitted."""
    prepared = review_diff.prepare_diff(_block("src/mesh.py", 1, 1), limit=10_000)
    assert prepared.dropped == []
    assert prepared.notes() == []


@pytest.mark.unit
def test_exclusion_pattern_matches_only_the_named_directory() -> None:
    """The pattern is a path match, not a name match."""
    assert review_diff.is_excluded(
        "benchmarks/results.jsonl", review_diff.EXCLUDED_PATTERNS
    )
    assert not review_diff.is_excluded("results.jsonl", review_diff.EXCLUDED_PATTERNS)
    assert not review_diff.is_excluded(
        "benchmarks/README.md", review_diff.EXCLUDED_PATTERNS
    )
