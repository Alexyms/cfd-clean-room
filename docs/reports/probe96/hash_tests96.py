"""Which test modules assert a stored hash, and whether this branch edits one.

    python docs/reports/probe96/hash_tests96.py

Writes ``results/builder96/hash_tests.json``: the merge base with main, the head
(HEAD's hash read from the checkout's files), every test module the branch
changes, the test modules at the base that hold a stored hash (a 64-digit
literal, ``sha256`` or ``hexdigest``) and the ones among them the branch edits.
Prediction (a) of prompt 96 holds only if the last list is empty (T6, T11).
"""

import json
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = ROOT / "results" / "builder96" / "hash_tests.json"
BASE_REF = "origin/main"
HASH = re.compile(r"[0-9a-f]{64}|sha256|hexdigest")


def git(*args: str) -> str:
    """The output of one git command in the checkout."""
    done = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    return done.stdout


def main() -> int:
    """Write the record."""
    head = git("rev-parse", "HEAD").strip()
    base = git("merge-base", BASE_REF, "HEAD").strip()
    changed = set(git("diff", "--name-only", f"{base}..{head}").split())
    edited_tests = sorted(
        name for name in changed if name.startswith("tests/") and name.endswith(".py")
    )
    modules = [
        name
        for name in git("ls-tree", "-r", "--name-only", base, "tests").split()
        if name.startswith("tests/test_") and name.endswith(".py")
    ]
    hash_tests = [
        name for name in modules if HASH.search(git("show", f"{base}:{name}"))
    ]
    record = {
        "base": base,
        "head": head,
        "edited_tests": edited_tests,
        "hash_tests": hash_tests,
        "hash_tests_edited": [name for name in hash_tests if name in changed],
    }
    OUT.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
