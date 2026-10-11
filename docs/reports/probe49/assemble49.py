"""Splice the generated tables into the report between their markers.

Usage:
    python assemble49.py            rewrite every marked table in the report
    python assemble49.py --check    exit 1 if any marked table is stale

Each table of tables49.py sits in docs/reports/ecr002_val017_annex20.md
between ``<!-- tables49 MODE begin -->`` and ``<!-- tables49 MODE end -->``.
The mode is run, its printed table captured, and the text between the
markers replaced, so the report's numbers are the records' and a rerun with
no new record is a zero-line diff. Everything outside the markers is kept
byte for byte, line endings included.
"""

import argparse
import contextlib
import io
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))

import tables49  # noqa: E402

REPORT = ROOT / "docs" / "reports" / "ecr002_val017_annex20.md"
MARK = re.compile(
    r"(<!-- tables49 (?P<mode>\w+) begin -->)(?P<body>.*?)(<!-- tables49 (?P=mode) end -->)",
    re.DOTALL,
)


def table_text(mode: str) -> str:
    """The markdown table one mode prints."""
    func = dict(tables49.MODES)[mode]
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        func(argparse.Namespace())
    return buffer.getvalue().strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    raw = REPORT.read_bytes().decode("utf-8")
    newline = "\r\n" if "\r\n" in raw else "\n"

    def splice(m: re.Match) -> str:
        table = table_text(m["mode"]).replace("\n", newline)
        return f"{m.group(1)}{newline}{table}{newline}{m.group(4)}"

    out = MARK.sub(splice, raw)
    modes = MARK.findall(raw)
    print(f"{len(modes)} marked tables: {', '.join(m[1] for m in modes)}")
    if args.check:
        if out != raw:
            print("stale")
            sys.exit(1)
        print("current")
        return
    if out != raw:
        REPORT.write_bytes(out.encode("utf-8"))
        print(f"rewritten {REPORT.relative_to(ROOT)}")
    else:
        print("unchanged")


if __name__ == "__main__":
    main()
