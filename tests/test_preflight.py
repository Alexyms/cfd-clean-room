"""scripts/preflight.py, CLAUDE.md's pre-commit checklist as one command (ADR-014 I).

The tests replace the command runner, so nothing here runs ruff or the
suite: they pin which commands the checklist runs, that --fix changes only
the format and lint steps, that every step runs after one fails, and that
the exit status is 1 exactly when a step failed.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import preflight


@pytest.mark.unit
class TestSteps:
    def test_the_check_runs_the_four_steps_and_changes_nothing(self) -> None:
        planned = preflight.steps(False, python="py")
        assert planned == [
            ("format", ["py", "-m", "ruff", "format", "--check", "."]),
            ("lint", ["py", "-m", "ruff", "check", "."]),
            ("system map", ["py", "scripts/gen_system_map.py", "--check"]),
            (
                "tests",
                ["py", "-m", "pytest", "tests/", "-q", "-m", "unit or integration"],
            ),
        ]

    def test_fix_changes_the_format_and_lint_steps_only(self) -> None:
        checked, fixed = preflight.steps(False, "py"), preflight.steps(True, "py")
        assert fixed[0] == ("format", ["py", "-m", "ruff", "format", "."])
        assert fixed[1] == ("lint", ["py", "-m", "ruff", "check", ".", "--fix"])
        assert fixed[2:] == checked[2:]

    def test_the_tools_run_under_this_interpreter_by_default(self) -> None:
        assert all(
            command[0] == sys.executable for _, command in preflight.steps(False)
        )


@pytest.mark.unit
class TestRun:
    def test_every_step_runs_after_one_fails(self) -> None:
        seen: list[list[str]] = []

        def run(command: list[str]) -> int:
            seen.append(command)
            return 1 if len(seen) == 1 else 0

        planned = preflight.steps(False, "py")
        results = preflight.run_steps(planned, run)
        assert seen == [command for _, command in planned]
        assert results == [("format", 1), ("lint", 0), ("system map", 0), ("tests", 0)]

    @pytest.mark.parametrize("failing", [None, "format", "tests"])
    def test_the_exit_status_is_one_exactly_when_a_step_failed(
        self, monkeypatch: pytest.MonkeyPatch, failing: str | None
    ) -> None:
        names = {tuple(command): name for name, command in preflight.steps(True)}

        def run(command: list[str]) -> int:
            return 2 if names[tuple(command)] == failing else 0

        monkeypatch.setattr(preflight, "_run_in_root", run)
        assert preflight.main(["--fix"]) == (0 if failing is None else 1)
