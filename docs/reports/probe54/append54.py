"""Item 7 of prompt 54: append the staggered-mgcg harness rows to benchmarks/results.jsonl.

Usage:
    python append54.py [--dry-run]

Reads the records of ``row54.py`` labelled ``mg`` (results/builder54/row_CASE_mg.json
of the main checkout), one per harness case, and appends the row each holds,
exactly as ``run_case`` returned it, to benchmarks/results.jsonl, in the
order of CASES. A row whose ``run_id`` is already in the file is skipped, so
a second run appends nothing. The staggered-cg rows of the same session
(label ``jacobi``) are parity checks and are not appended. Every appended row
must carry the method ``staggered-mgcg`` and a clean commit; anything else is
an error before any line is written.
"""

import argparse
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parents[3]
MAIN = Path(
    subprocess.run(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
        cwd=HERE,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
).parent
OUT = MAIN / "results" / "builder54"
RESULTS = HERE / "benchmarks" / "results.jsonl"
CASES = (
    "val001_80x40",
    "val001_80x40_stretched",
    "val002_20x20",
    "val002_40x40",
    "val002_80x80",
    "val017_216x72",
    "val019_656x60",
)


def main() -> None:
    """Check every row, then append the ones not yet in the file."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    rows = []
    for case in CASES:
        row = json.loads((OUT / f"row_{case}_mg.json").read_text())["row"]
        if row["method"] != "staggered-mgcg":
            raise SystemExit(f"{case}: method {row['method']!r}")
        if row["git_dirty"]:
            raise SystemExit(f"{case}: run from a dirty tree")
        rows.append(row)
    commits = {row["git_commit"] for row in rows}
    if len(commits) != 1:
        raise SystemExit(f"rows come from more than one commit: {sorted(commits)}")
    raw = RESULTS.read_bytes()
    newline = b"\r\n" if b"\r\n" in raw else b"\n"
    stored = {
        json.loads(line)["run_id"]
        for line in raw.decode("utf-8").splitlines()
        if line.strip()
    }
    new = [row for row in rows if row["run_id"] not in stored]
    print(f"{len(new)} of {len(rows)} rows to append (commit {commits.pop()[:7]})")
    if args.dry_run or not new:
        return
    # The file keeps the line ending it has, so the diff is the appended lines.
    if not raw.endswith(b"\n"):
        raw += newline
    with RESULTS.open("wb") as handle:
        handle.write(raw)
        for row in new:
            handle.write(json.dumps(row).encode("utf-8") + newline)


if __name__ == "__main__":
    main()
