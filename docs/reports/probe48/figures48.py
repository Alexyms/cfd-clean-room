"""The figures of prompt 48, drawn by prompt 47's figure code.

Usage:
    python figures48.py all        every converged flow record and every march
    python figures48.py publish    copy the new figures beside the report

`all` runs docs/reports/probe47/figures47.py's `all` on results/builder48/,
drawing into results/builder48/figures/ (the three-panel flow figure of
every converged run, the concentration figure of every march, the room's
boundaries marked). `publish` copies each figure to docs/reports/ under its
own name, ecr002_step6_<record>.png. A figure whose name is already there
(prompt 47's upwind arm, which item 4 reproduces) is compared byte for byte
instead and left as it is; the comparison is printed and kept in
results/builder48/figures.json.
"""

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "docs" / "reports" / "probe47"))

import figures47  # noqa: E402

OUT = ROOT / "results" / "builder48"
DRAWN = OUT / "figures"
PUBLISHED = ROOT / "docs" / "reports"
figures47.OUT = OUT
figures47.FIGURES = DRAWN
FLOW = re.compile(r"^(standard|rng)_\d+x\d+_(upwind|umist)$")
MARCH = re.compile(r"^transport_(standard|rng)_\d+x\d+_(upwind|umist)_S\d$")


def draw(_args: argparse.Namespace) -> None:
    """Every figure into results/builder48/figures/.

    figures47's own `all` reads every JSON file in the directory; this
    build's directory also holds the laminar, plant and test records, so
    the flow and march records are picked by name here.
    """
    DRAWN.mkdir(parents=True, exist_ok=True)
    for path in sorted(OUT.glob("*.json")):
        if not FLOW.match(path.stem):
            continue
        rec = json.loads(path.read_text())
        if rec.get("stop") == "error_estimate_and_continuity" and "core" in rec:
            figures47.fields(argparse.Namespace(name=path.stem))
    for path in sorted(OUT.glob("transport_*.json")):
        if MARCH.match(path.stem):
            figures47.concentration(argparse.Namespace(label=path.stem))


def publish(_args: argparse.Namespace) -> None:
    """New figures beside the report; existing names compared, not replaced."""
    record = {}
    for path in sorted(DRAWN.glob("ecr002_step6_*.png")):
        target = PUBLISHED / path.name
        if target.exists():
            same = target.read_bytes() == path.read_bytes()
            record[path.name] = "exists, identical" if same else "exists, differs"
        else:
            shutil.copy2(path, target)
            record[path.name] = "published"
    (OUT / "figures.json").write_text(json.dumps(record, indent=1))
    print(json.dumps(record, indent=1))


def main() -> None:
    """Dispatch one of the modes the module docstring lists."""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    sub.add_parser("all").set_defaults(func=draw)
    sub.add_parser("publish").set_defaults(func=publish)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
