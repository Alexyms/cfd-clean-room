"""Prepare a pull request diff for the reviewer.

Three things happen to the raw `gh pr diff` output before it reaches the
model, and each exists because of a failure that was measured on PR #13 and
PR #14.

1. Generated data files are excluded. `benchmarks/results.jsonl` is
   append-only and grows on every PR; on #13 it was 130,083 of 157,467
   characters, on #14 163,363 of 219,618. It sorts first, so under a
   character budget it consumed the whole budget and the reviewer saw no
   source at all. Each excluded file is replaced by one summary line naming
   it and the number of lines added and removed, so the reviewer still
   learns that the numbers moved and can ask for them.

2. The remaining diff is bounded to a character limit, file by file. A file
   is either included whole or not at all; files are taken in the order the
   diff lists them and a file that does not fit is skipped rather than cut,
   so a later, smaller file can still make it in.

3. When the limit bites, the reviewer is told which paths were dropped, not
   only that truncation happened. "This diff was truncated" cannot be acted
   on; "scripts/benchmark.py was dropped" can.

Standard library only, so the test suite can import it without the API
clients that review.py needs.
"""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass, field

# Generated data that is appended on every PR. A list so the next case can be
# added without touching the logic. Patterns are matched against the full
# repository-relative path with fnmatch semantics.
EXCLUDED_PATTERNS: list[str] = [
    "benchmarks/*.jsonl",
]

DEFAULT_CHAR_LIMIT = 100_000

_FILE_HEADER = re.compile(r"^diff --git a/(?P<a>.+?) b/(?P<b>.+)$", re.MULTILINE)


@dataclass
class FileDiff:
    """One file's block of a unified diff, with its line counts."""

    path: str
    text: str
    added: int
    removed: int

    @property
    def chars(self) -> int:
        """Size of the block in characters, the unit the limit is set in."""
        return len(self.text)


@dataclass
class PreparedDiff:
    """What the reviewer receives, plus what was taken away and why."""

    text: str
    kept: list[FileDiff] = field(default_factory=list)
    excluded: list[FileDiff] = field(default_factory=list)
    dropped: list[FileDiff] = field(default_factory=list)
    limit: int = DEFAULT_CHAR_LIMIT

    @property
    def truncated(self) -> bool:
        """True when at least one non-excluded file did not fit the limit."""
        return bool(self.dropped)

    def notes(self) -> list[str]:
        """Lines to place above the diff, one per excluded or dropped file."""
        lines: list[str] = []
        for entry in self.excluded:
            lines.append(
                f"Excluded generated data file `{entry.path}`: "
                f"+{entry.added} lines, -{entry.removed} lines. "
                "The values are not in this diff; ask for them if they matter."
            )
        if self.dropped:
            names = ", ".join(
                f"`{entry.path}` ({entry.chars:,} chars)" for entry in self.dropped
            )
            lines.append(
                f"Diff truncated to {self.limit:,} characters. "
                f"These files were DROPPED IN FULL and are not visible below: {names}. "
                "Do not report on them as absent from the change; they are absent from this view."
            )
        return lines


def split_diff(diff: str) -> list[FileDiff]:
    """Split a unified diff into per-file blocks in the order they appear.

    Line counts come from the hunks only. The `---`/`+++` header lines sit
    before the first `@@` and are skipped, so a removed line whose content
    begins with `--` is not mistaken for a header.
    """
    headers = list(_FILE_HEADER.finditer(diff))
    blocks: list[FileDiff] = []
    for index, match in enumerate(headers):
        start = match.start()
        end = headers[index + 1].start() if index + 1 < len(headers) else len(diff)
        text = diff[start:end]
        added = removed = 0
        in_hunk = False
        for line in text.splitlines():
            if line.startswith("@@"):
                in_hunk = True
                continue
            if not in_hunk:
                continue
            if line.startswith("+"):
                added += 1
            elif line.startswith("-"):
                removed += 1
        blocks.append(FileDiff(match.group("b"), text, added, removed))
    return blocks


def is_excluded(path: str, patterns: list[str]) -> bool:
    """True when the path matches any exclusion pattern."""
    return any(fnmatch.fnmatchcase(path, pattern) for pattern in patterns)


def prepare_diff(
    diff: str,
    limit: int = DEFAULT_CHAR_LIMIT,
    patterns: list[str] | None = None,
) -> PreparedDiff:
    """Exclude generated data, bound the size, and record what was removed.

    Parameters
    ----------
    diff : str
        Raw unified diff, as `gh pr diff` prints it.
    limit : int
        Maximum characters of diff text handed to the reviewer.
    patterns : list[str] | None
        Exclusion patterns; defaults to EXCLUDED_PATTERNS.

    Returns
    -------
    PreparedDiff
        The text to review and the excluded and dropped files. Files that
        are neither excluded nor dropped are passed through byte for byte.
    """
    patterns = EXCLUDED_PATTERNS if patterns is None else patterns
    prepared = PreparedDiff(text="", limit=limit)
    used = 0
    for block in split_diff(diff):
        if is_excluded(block.path, patterns):
            prepared.excluded.append(block)
        elif used + block.chars <= limit:
            prepared.kept.append(block)
            used += block.chars
        else:
            prepared.dropped.append(block)
    prepared.text = "".join(block.text for block in prepared.kept)
    return prepared
