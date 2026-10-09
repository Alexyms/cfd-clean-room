"""Laminar checks of prompt 45 (ECR-002 step 6), run at every code commit.

Usage:
    python check45.py [--root TREE] val CASE --label L    a VAL preset's stop, count and face hash
    python check45.py [--root TREE] ladder RUNG --label L  the 40x15 ladder at Re 895 or 8950
    python check45.py report --label L                     every record of a label against step 4's

The measuring is step 4's, unchanged: docs/reports/probe43/field43.py's ``val``
and ``ladder`` (which reuse docs/reports/probe41/outlet41.py's runner and face
hash, SHA-256 over u's bytes then v's), imported from TREE so that a worktree
of another commit measures that commit. Records go to results/builder45/ of
the main checkout; the references are step 4's round-2 records,
results/builder43/demo_43b/, taken at the base of this branch on this machine
(``val_*_43b`` and ``ladder_43b_*``).
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
OUT = MAIN / "results" / "builder45"
REFERENCE = MAIN / "results" / "builder43" / "demo_43b"
CASES = ("val001_80x40", "val001_80x40_stretched", "val002_80x80")
RUNGS = ("895", "8950")


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


def report(label: str) -> None:
    """Stop, count and face hash of each record of LABEL against step 4's."""
    rows = []
    for case in CASES:
        rows.append((f"val_{case}_{label}", f"val_{case}_43b"))
    for rung in RUNGS:
        rows.append((f"ladder_{label}_{rung}", f"ladder_43b_{rung}"))
    summary = []
    for name, ref in rows:
        path = OUT / f"{name}.json"
        if not path.exists():
            summary.append({"record": name, "present": False})
            continue
        built = json.loads(path.read_text())
        base = json.loads((REFERENCE / f"{ref}.json").read_text())
        summary.append(
            {
                "record": name,
                "stop": built["stop"],
                "outer": built["outer"],
                "face_hash": built["face_hash"][:8],
                "equal_to_step4": (
                    built["stop"] == base["stop"]
                    and built["outer"] == base["outer"]
                    and built["face_hash"] == base["face_hash"]
                ),
            }
        )
    text = json.dumps(summary, indent=1)
    (OUT / f"report_{label}.json").write_text(text)
    print(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(HERE))
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("val")
    p.add_argument("case", choices=CASES)
    p.add_argument("--label", required=True)
    p = sub.add_parser("ladder")
    p.add_argument("rung", choices=RUNGS)
    p.add_argument("--label", required=True)
    p = sub.add_parser("report")
    p.add_argument("--label", required=True)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.command == "report":
        report(args.label)
        return
    field43 = load_field43(Path(args.root))
    if args.command == "val":
        field43.val(argparse.Namespace(case=args.case, zero=False, label=args.label))
    else:
        field43.ladder(argparse.Namespace(rung=args.rung, label=args.label))


if __name__ == "__main__":
    main()
