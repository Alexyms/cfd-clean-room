"""Insert the generated tables into the step 6 comparative report.

Usage:
    python assemble47.py

Reads results/builder47/tables_all.md (the output of `run47.sh tables`,
one `### MODE` heading per table set) and the geometry table of rooms47.py,
and replaces, in docs/reports/ecr002_step6_comparative.md, the text between
each pair of marker lines

    <!-- tables47 MODE begin -->
    ...
    <!-- tables47 MODE end -->

with that mode's generated tables, verbatim. The prose is the report's and
is not touched; a regenerated table replaces the old one by running this
script again. MODE `rooms` takes the geometry table from
results/builder47/rooms.md.
"""

import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = ROOT / "results" / "builder47"
REPORT = ROOT / "docs" / "reports" / "ecr002_step6_comparative.md"


def tables() -> dict[str, str]:
    """The generated tables, keyed by tables47's mode name."""
    text = (OUT / "tables_all.md").read_text()
    parts = re.split(r"^### (\w+)\n", text, flags=re.M)
    out = {parts[i]: parts[i + 1].strip() for i in range(1, len(parts), 2)}
    rooms = OUT / "rooms.md"
    if rooms.exists():
        out["rooms"] = rooms.read_text().strip()
    return out


def main() -> None:
    t = tables()
    text = REPORT.read_text()
    pattern = re.compile(
        r"(<!-- tables47 (\w+) begin -->)\n.*?(<!-- tables47 \2 end -->)", re.S
    )
    missing: list[str] = []

    def replace(m: re.Match) -> str:
        mode = m.group(2)
        if mode not in t:
            missing.append(mode)
            return m.group(0)
        return f"{m.group(1)}\n{t[mode]}\n{m.group(3)}"

    new, n = pattern.subn(replace, text)
    REPORT.write_text(new)
    print(f"replaced {n - len(missing)} table blocks; missing {missing or 'none'}")


if __name__ == "__main__":
    main()
