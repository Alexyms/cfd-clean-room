"""Laminar checks of prompt 50 (item 3): VAL-001, its stretched twin and VAL-002 hash to the base.

Usage:
    python check50.py [--root TREE] val CASE --label L    a VAL preset's stop, count and face hash
    python check50.py report --base B --built L            each case's record of L against B's

The measuring is step 4's, unchanged: docs/reports/probe43/field43.py's ``val``
(the committed solver on the preset, SHA-256 over u's bytes then v's),
imported from TREE so that a worktree of another commit measures that commit.
The base is measured in a worktree of origin/main at the branch point, the
built tree in the branch's own checkout at commit A, on one machine. Records go to
results/builder50/ of the main checkout as val_CASE_LABEL.json.
"""

import argparse
import importlib
import json
import subprocess
import sys
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
OUT = MAIN / "results" / "builder50"
CASES = ("val001_80x40", "val001_80x40_stretched", "val002_80x80")


def load_field43(root: Path) -> object:
    """field43 from TREE, its records redirected to OUT."""
    root = root.resolve()
    sys.path.insert(0, str(root / "docs" / "reports" / "probe43"))
    field43 = importlib.import_module("field43")
    if Path(field43.__file__).resolve().parents[3] != root:
        raise SystemExit(f"field43 was imported from {field43.__file__}, not {root}")
    field43.OUT = OUT
    field43.probe.OUT = OUT
    return field43


def report(base: str, built: str) -> None:
    """Stop, count and face hash of each case under BUILT against BASE."""
    summary = []
    for case in CASES:
        paths = {lab: OUT / f"val_{case}_{lab}.json" for lab in (base, built)}
        if not all(p.exists() for p in paths.values()):
            summary.append({"case": case, "present": False})
            continue
        a, b = (json.loads(paths[lab].read_text()) for lab in (base, built))
        summary.append(
            {
                "case": case,
                "stop": (a["stop"], b["stop"]),
                "outer": (a["outer"], b["outer"]),
                "face_hash": (a["face_hash"][:16], b["face_hash"][:16]),
                "equal": a["stop"] == b["stop"]
                and a["outer"] == b["outer"]
                and a["face_hash"] == b["face_hash"],
            }
        )
    text = json.dumps(summary, indent=1)
    (OUT / f"report_{base}_{built}.json").write_text(text)
    print(text)


def main() -> None:
    """Dispatch one of the modes the module docstring lists."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(HERE))
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("val")
    p.add_argument("case", choices=CASES)
    p.add_argument("--label", required=True)
    p = sub.add_parser("report")
    p.add_argument("--base", required=True)
    p.add_argument("--built", required=True)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.command == "report":
        report(args.base, args.built)
        return
    field43 = load_field43(Path(args.root))
    field43.val(argparse.Namespace(case=args.case, zero=False, label=args.label))


if __name__ == "__main__":
    main()
