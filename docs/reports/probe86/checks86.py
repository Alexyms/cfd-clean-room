"""Record the lint job's checks and the tree they run on (prompt 86b).

    python docs/reports/probe86/checks86.py

Run with Windows' ``.venv``, with nothing else changing the checkout. It runs
the three commands of the lint job (``ruff format --check .``, ``ruff check .``
and ``scripts/gen_system_map.py --check``) and writes the commit at ``HEAD``,
the files that differ from it and each command's exit code and last line to
``results/builder86/checks.json``, which ``tables86.py`` reads beside the two
suites' JUnit files. The suites and these checks are run on the same tree, so a
record that names uncommitted files names the edits the next commit holds.
``tables86.py --check`` is not recorded here: it compares the report with this
record and the others, so it is run by whoever reads the report.
"""

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = ROOT / "results" / "builder86" / "checks.json"
COMMANDS = {
    "ruff format --check .": [sys.executable, "-m", "ruff", "format", "--check", "."],
    "ruff check .": [sys.executable, "-m", "ruff", "check", "."],
    "python scripts/gen_system_map.py --check": [
        sys.executable,
        "scripts/gen_system_map.py",
        "--check",
    ],
}


def git(*args: str) -> str:
    """Return the standard output of a git command run in the checkout."""
    done = subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
    )
    return done.stdout


def run(command: list[str]) -> dict:
    """Run one check and return its exit code and the last line it printed."""
    done = subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8"
    )
    lines = [line for line in (done.stdout + done.stderr).splitlines() if line.strip()]
    return {"returncode": done.returncode, "last_line": lines[-1] if lines else ""}


def main() -> int:
    """Run the checks and write the record."""
    uncommitted = [line[3:] for line in git("status", "--porcelain").splitlines()]
    record = {
        "head": git("rev-parse", "HEAD").strip(),
        "uncommitted": uncommitted,
        "checks": {name: run(command) for name, command in COMMANDS.items()},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(record, indent=1), encoding="utf-8")
    for name, result in record["checks"].items():
        print(f"{name}: exit {result['returncode']}: {result['last_line']}")
    print("wrote", OUT)
    return 0 if all(r["returncode"] == 0 for r in record["checks"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())
