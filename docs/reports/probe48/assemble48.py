"""Insert the generated tables into the step 6 variant report.

Usage:
    python assemble48.py

Reads results/builder48/tables_all.md (the output of `run48.sh tables`,
one `### MODE` heading per table set), and replaces, in
docs/reports/ecr002_step6_variants.md, the text between each pair of
marker lines

    <!-- tables48 MODE begin -->
    ...
    <!-- tables48 MODE end -->

with that mode's generated tables, verbatim. The prose is the report's and
is not touched; a regenerated table replaces the old one by running this
script again.
"""

import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = ROOT / "results" / "builder48"
REPORT = ROOT / "docs" / "reports" / "ecr002_step6_variants.md"


def tables() -> dict[str, str]:
    """The generated tables, keyed by tables48's mode name."""
    text = (OUT / "tables_all.md").read_text()
    parts = re.split(r"^### (\w+)\n", text, flags=re.M)
    out = {parts[i]: parts[i + 1].strip() for i in range(1, len(parts), 2)}
    return out


def main() -> None:
    """Replace every marked block with its generated tables."""
    t = tables()
    text = REPORT.read_text()
    pattern = re.compile(
        r"(<!-- tables48 (\w+) begin -->)\n.*?(<!-- tables48 \2 end -->)", re.S
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
