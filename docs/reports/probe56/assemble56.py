"""Insert the generated tables into the step 8 product report.

Usage:
    python assemble56.py

Reads results/builder56/tables_all.md (the output of `run56.sh tables`,
one `### MODE` heading per table set) and replaces, in
docs/reports/ecr002_step8_product.md, the text between each pair of
marker lines

    <!-- tables56 MODE begin -->
    ...
    <!-- tables56 MODE end -->

with that mode's generated tables, verbatim. The prose is the report's and
is not touched; a regenerated table replaces the old one by running this
script again. The report's line endings are kept as found.
"""

import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = ROOT / "results" / "builder56"
REPORT = ROOT / "docs" / "reports" / "ecr002_step8_product.md"


def tables() -> dict[str, str]:
    """The generated tables, keyed by tables56's mode name."""
    text = (OUT / "tables_all.md").read_text(encoding="utf-8")
    parts = re.split(r"^### (\w+)\n", text, flags=re.M)
    return {parts[i]: parts[i + 1].strip() for i in range(1, len(parts), 2)}


def main() -> None:
    t = tables()
    raw = REPORT.read_bytes().decode("utf-8")
    crlf = "\r\n" in raw
    text = raw.replace("\r\n", "\n")
    pattern = re.compile(
        r"(<!-- tables56 (\w+) begin -->)\n.*?(<!-- tables56 \2 end -->)", re.S
    )
    missing: list[str] = []

    def replace(m: re.Match) -> str:
        mode = m.group(2)
        if mode not in t:
            missing.append(mode)
            return m.group(0)
        return f"{m.group(1)}\n{t[mode]}\n{m.group(3)}"

    new, n = pattern.subn(replace, text)
    if crlf:
        new = new.replace("\n", "\r\n")
    REPORT.write_bytes(new.encode("utf-8"))
    print(f"replaced {n - len(missing)} table blocks; missing {missing or 'none'}")


if __name__ == "__main__":
    main()
