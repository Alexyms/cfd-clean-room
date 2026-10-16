#!/usr/bin/env python
"""CLAUDE.md's pre-commit checklist in one command (ADR-014 section I).

Run from the repository root with the project's interpreter:

    python scripts/preflight.py           # check: changes nothing
    python scripts/preflight.py --fix     # format and fix lint first, then check

The steps, in order: ``ruff format --check`` (``ruff format`` with --fix),
``ruff check`` (``ruff check --fix`` with --fix), the system map's
``--check`` (``scripts/gen_system_map.py``), and the unit and integration
tiers of the suite (``pytest tests/ -m "unit or integration"``). The
validation tier is left to the full suite, which takes far longer and which
CLAUDE.md asks for before a merge, not before every commit.

Every step runs even after one fails, so one run lists every failure; the
exit status is 1 when any step failed and 0 otherwise. Each tool runs with
the interpreter that runs this script (``python -m ruff``, ``python -m
pytest``), so the virtual environment's versions are the ones checked.
"""

import argparse
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUITE_TIERS = "unit or integration"


def steps(fix: bool, python: str = sys.executable) -> list[tuple[str, list[str]]]:
    """The checklist as (name, command) pairs, in the order they run.

    Parameters
    ----------
    fix : bool
        True replaces the format and lint checks by the commands that fix
        what they can; the system map and the tests are checks either way.
    python : str, optional
        The interpreter each tool runs under; this script's by default.

    Returns
    -------
    list[tuple[str, list[str]]]
        Four steps: format, lint, the system map and the two test tiers.
    """
    if fix:
        format_step = [python, "-m", "ruff", "format", "."]
        lint_step = [python, "-m", "ruff", "check", ".", "--fix"]
    else:
        format_step = [python, "-m", "ruff", "format", "--check", "."]
        lint_step = [python, "-m", "ruff", "check", "."]
    return [
        ("format", format_step),
        ("lint", lint_step),
        ("system map", [python, "scripts/gen_system_map.py", "--check"]),
        ("tests", [python, "-m", "pytest", "tests/", "-q", "-m", SUITE_TIERS]),
    ]


def run_steps(
    planned: Sequence[tuple[str, list[str]]],
    run: Callable[[list[str]], int],
) -> list[tuple[str, int]]:
    """Run every step, whatever the earlier ones returned.

    Parameters
    ----------
    planned : Sequence[tuple[str, list[str]]]
        The (name, command) pairs of ``steps``.
    run : Callable[[list[str]], int]
        Runs one command and returns its exit status.

    Returns
    -------
    list[tuple[str, int]]
        Each step's name and exit status, in order.
    """
    return [(name, run(command)) for name, command in planned]


def _run_in_root(command: list[str]) -> int:
    """Run a command from the repository root with its output shown."""
    return subprocess.run(command, cwd=ROOT, check=False).returncode


def main(argv: Sequence[str] | None = None) -> int:
    """Run the checklist and print one line per step.

    Parameters
    ----------
    argv : Sequence[str], optional
        The command-line arguments; ``sys.argv[1:]`` when None.

    Returns
    -------
    int
        1 when any step failed, 0 otherwise.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--fix", action="store_true", help="format and fix lint before checking"
    )
    args = parser.parse_args(argv)
    results = run_steps(steps(args.fix), _run_in_root)
    print()
    for name, status in results:
        print(f"{name:<11} {'ok' if status == 0 else f'FAILED ({status})'}")
    return 1 if any(status != 0 for _, status in results) else 0


if __name__ == "__main__":
    sys.exit(main())
