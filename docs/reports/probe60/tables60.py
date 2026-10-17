"""The tables of docs/reports/adr014_step0.md, from the records of prompt 60.

Usage:
    python tables60.py           # rewrite the marked tables in the report
    python tables60.py --check   # exit 1 if a marked table differs

Reads results/builder60/ of the main checkout (hash_*.json from hash60.py,
warm60_*.json from warm60.py, plants_a.json and plants_b.json from the
planted-defect runs) and the records the hashes of record come from:
results/builder54/ (prompt 54: the harness rows and the product rows under
the multigrid default, docs/reports/multigrid_default.md's hash table),
results/builder56/L0_320x120.json (ECR-002 step 8) and results/builder59/
(prompt 59's T1). Each table sits in the report between
``<!-- BEGIN tables60: NAME -->`` and ``<!-- END tables60: NAME -->``.
"""

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import hash60  # noqa: E402

REPORT = HERE.parent / "adr014_step0.md"

# (label in results/builder60, the record holding the hash of record, the
# record's own name for it)
CASES = [
    ("val001_80x40", "builder54/row_val001_80x40_mg.json", "prompt 54"),
    (
        "val001_80x40_stretched",
        "builder54/row_val001_80x40_stretched_mg.json",
        "prompt 54",
    ),
    ("val002_20x20", "builder54/row_val002_20x20_mg.json", "prompt 54"),
    ("val002_40x40", "builder54/row_val002_40x40_mg.json", "prompt 54"),
    ("val002_80x80", "builder54/row_val002_80x80_mg.json", "prompt 54"),
    ("val017_216x72", "builder54/row_val017_216x72_mg.json", "prompt 54"),
    ("val019_656x60", "builder54/row_val019_656x60_mg.json", "prompt 54"),
    ("product_80x30", "builder54/product_80x30_mg.json", "prompt 54"),
    ("product_160x60", "builder54/product_160x60_mg.json", "prompt 54"),
    ("L0_320x120", "builder56/L0_320x120.json", "ECR-002 step 8"),
    ("t1_rest", "builder59/rest.json", "prompt 59, T1 rest arm"),
]


def load(path: Path) -> dict | None:
    """A JSON record, or None when it is not there."""
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def record_stop(record: dict) -> tuple[str, int]:
    """The stop and the outer count a kept record states, whichever writer made it."""
    if "row" in record:
        row = record["row"]
        return row["outcome"]["stop_reason"], row["work"]["outer_iterations"]
    return record["stop"], record["outer"]


def table(headers: list[str], rows: list[list[str]]) -> str:
    """A markdown table, each cell's own pipes escaped."""
    escape = chr(92) + "|"
    lines = ["| " + " | ".join(h.replace("|", escape) for h in headers) + " |"]
    lines.append("|" + "---|" * len(headers))
    lines += ["| " + " | ".join(c.replace("|", escape) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def hashes(results: Path) -> str:
    """Prediction (a): each case from rest under the branch against its hash of record."""
    rows = []
    for label, source, origin in CASES:
        kept = load(results / source)
        run = load(results / "builder60" / f"hash_{label}.json")
        base = load(results / "builder60" / f"hash_{label}_base.json")
        if kept is None:
            raise SystemExit(f"no record of {label} at results/{source}")
        stop, outer = record_stop(kept)
        commit = kept.get("commit", "?")
        of_record = f"{kept['face_hash'][:16]} ({commit}, {origin})"
        if run is None:
            rows.append(
                [label, of_record, f"{stop} at {outer:,}", "not run", "-", "-", "-"]
            )
            continue
        same = "yes" if run["face_hash"] == kept["face_hash"] else "NO"
        control = "-"
        if base is not None:
            control = f"{base['face_hash'][:16]} ({base['commit']})"
        rows.append(
            [
                label,
                of_record,
                f"{stop} at {outer:,}",
                f"{run['face_hash'][:16]} ({run['commit']})",
                f"{run['stop']} at {run['outer']:,}",
                same,
                control,
            ]
        )
    return table(
        [
            "Case",
            "Hash of record (commit, source)",
            "Its stop",
            "From rest, initial=None (commit)",
            "Stop",
            "Equal",
            "Base 1c58700 control",
        ],
        rows,
    )


def warm(results: Path) -> str:
    """Item 6: the two warm-started product solves."""
    rows = []
    t1 = load(results / "builder59" / "warm.json")
    for arm, door in (("self", "closed"), ("open", "open")):
        run = load(results / "builder60" / f"warm60_{arm}.json")
        if run is None:
            rows.append([arm, door, "not run", "-", "-", "-", "-", "-"])
            continue
        change = run["largest_face_change_from_start"]
        moved = f"{change['u']:.3e}, {change['v']:.3e}"
        if arm == "self":
            bound = run["iteration_error_tol"] * run["velocity_scale"]
            moved += f" (bound {bound:.2e})"
        if arm == "open" and t1 is not None:
            against = (
                f"T1 warm {t1['outer']:,}; outer "
                f"{'equal' if run['matches_t1']['outer'] else 'DIFFERS'}, faces "
                f"{'bitwise' if run['matches_t1']['face_hash'] else 'DIFFER'}, history "
                f"{'equal' if run['matches_t1']['residual_history'] else 'DIFFERS'}"
            )
        else:
            against = f"within one window: {'yes' if run['outer'] <= 100 else 'NO'}"
        rows.append(
            [
                arm,
                door,
                f"{run['stop']} at {run['outer']:,}",
                against,
                moved,
                run["face_hash"][:16],
                f"{run['wall_seconds']:.0f}",
                str(run["concurrent_processes"]),
            ]
        )
    return table(
        [
            "Arm",
            "Door",
            "Stop",
            "Against the prediction",
            "Largest face change from L0, u, v (m/s)",
            "Face hash",
            "Wall (s)",
            "Processes",
        ],
        rows,
    )


def plants(results: Path) -> str:
    """The planted defects, each with its control and its result."""
    rows = []
    for name in ("plants_a.json", "plants_b.json"):
        for p in load(results / "builder60" / name) or []:
            rows.append(
                [
                    p["plant"],
                    p["file"],
                    ", ".join(Path(t).name for t in p["tests"]),
                    str(p["control"]),
                    str(p["planted"]),
                    p["verdict"],
                ]
            )
    return table(
        ["Planted defect", "File", "Tests", "Control exit", "Planted exit", "Result"],
        rows,
    )


def render(results: Path) -> dict[str, str]:
    """Every table, by its marker name."""
    return {"hashes": hashes(results), "warm": warm(results), "plants": plants(results)}


def splice(text: str, tables: dict[str, str]) -> str:
    """The report with each marked table replaced."""
    for name, body in tables.items():
        pattern = re.compile(
            rf"(<!-- BEGIN tables60: {name} -->\n).*?(<!-- END tables60: {name} -->)",
            re.S,
        )
        text, n = pattern.subn(
            lambda m, b=body: m.group(1) + b + "\n" + m.group(2), text
        )
        if n != 1:
            raise SystemExit(f"marker pair for {name} found {n} times in {REPORT}")
    return text


def main() -> None:
    """Rewrite or check the report's tables."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    results = hash60.main_results().parent
    raw = REPORT.read_bytes()
    crlf = b"\r\n" in raw
    text = raw.decode("ascii").replace("\r\n", "\n")
    new = splice(text, render(results))
    if args.check:
        if new != text:
            print("the report's tables differ from the records")
            sys.exit(1)
        print("the report's tables match the records")
        return
    REPORT.write_bytes((new.replace("\n", "\r\n") if crlf else new).encode("ascii"))
    print("tables written")


if __name__ == "__main__":
    main()
