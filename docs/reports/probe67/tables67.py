"""The tables of ADR-014's step 6a, generated from the prompt 67 records.

    python tables67.py            # writes docs/reports/probe67/tables67.md
    python tables67.py --check    # regenerates and compares with the committed file

Reads results/builder67/order67.json (order67.py), results/builder67/smoke67.json
(smoke67.py), results/builder59/cylinder_A.json and cylinder_B.json (T2's arms
A and B) and the hash records results/builder62/hash_*.json (hash62.py; the
suffix _67 is this step's source, _64 prompt 64's and no suffix prompt 62's).
A record that is missing prints as "not run". The ADR copies the tables from
the written file and points here; the prose beside them states no number of
its own.
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RESULTS = ROOT / "results" / "builder67"
PROBE59 = ROOT / "results" / "builder59"
HASHES = ROOT / "results" / "builder62"
TABLES = HERE / "tables67.md"
HASH_CASES = (
    "val001_80x40",
    "val001_80x40_stretched",
    "val002_20x20",
    "val002_40x40",
    "val002_80x80",
    "val017_216x72",
    "val019_656x60",
    "product_80x30",
    "product_160x60",
    "L0_320x120",
    "t1_rest",
)


def load(path: Path) -> dict | None:
    """The JSON record at PATH as a mapping, or None when it was not written."""
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def order_tables() -> list[str]:
    """O1 (the ladder) and O2 (the iteration error against the time differences)."""
    rec = load(RESULTS / "order67.json")
    lines = [
        "### O1. The observed order in time: the cavity spun up from rest",
        "",
    ]
    if rec is None:
        return [*lines, "not run", ""]
    lines += [
        f"VAL-002's cavity at Re 100 on a fixed uniform {rec['grid']}x{rec['grid']} "
        f"grid, from rest to t = {rec['t_end']:g} s; the field is {rec['field']}. "
        f"Per-step criterion {rec['step_tol']:g} of the lid speed, pressure_rtol "
        f"{rec['pressure_rtol']:g}. Commit {rec['commit']}. The order on a row is "
        "log2 of the ratio of the difference on the row above to the difference on "
        "the row; the asserted order is the last row's (tests/test_time_order.py).",
        "",
        "| Scheme | dt (s) | Steps | Outer per step: mean, max | Worst imbalance (kg/s per m) | Wall (s) | Difference to dt/2 (m/s) | Observed order |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for scheme, data in rec["schemes"].items():
        diffs = data["differences"]
        orders = data["orders"]
        for i, run in enumerate(data["runs"]):
            diff = f"{diffs[i]:.3e}" if i < len(diffs) else "-"
            order = f"{orders[i - 1]:.3f}" if 1 <= i <= len(orders) else "-"
            lines.append(
                f"| {scheme} | {run['dt']:g} | {run['steps']} | {run['outer_mean']:.1f}, "
                f"{run['outer_max']} | {run['worst_imbalance_max']:.1e} | "
                f"{run['wall_seconds']:.1f} | {diff} | {order} |"
            )
    smallest = min(d for data in rec["schemes"].values() for d in data["differences"])
    lines += [
        "",
        "### O2. The iteration error the ladder's states carry",
        "",
        f"Each scheme's coarsest and finest runs repeated at a per-step criterion of "
        f"{rec['reference_tol']:g}; the iteration error is the largest face "
        "difference between the run and its repeat. The smallest time difference "
        "is the smallest entry of O1's difference column.",
        "",
        "| Scheme | dt (s) | Iteration error (m/s) | Smallest time difference (m/s) | Ratio | Repeat's outer per step | Repeat's wall (s) |",
        "|---|---|---|---|---|---|---|",
    ]
    for scheme, data in rec["schemes"].items():
        for dt, it in data["iteration_error"].items():
            lines.append(
                f"| {scheme} | {dt} | {it['error']:.2e} | {smallest:.2e} | "
                f"{it['error'] / smallest:.1e} | {it['reference_outer_mean']:.1f} | "
                f"{it['reference_wall_seconds']:.1f} |"
            )
    return [*lines, ""]


def smoke_tables() -> list[str]:
    """S1 (the arms against T2) and S2 (the implicit Euler arm against the probe's)."""
    rec = load(RESULTS / "smoke67.json")
    lines = ["### S1. The cylinder smoke run: outer iterations per step", ""]
    if rec is None:
        return [*lines, "not run", ""]
    start = rec["startup"]
    lines += [
        f"cylinder59.py's configuration on its uniform {rec['dx']:g} d grid "
        f"({rec['grid'][0]}x{rec['grid'][1]}), Re 100, from the start-up field "
        f"({start['outer']} steady outer iterations from rest; equal face for face "
        f"to the probe's kept field: {start['equal_to_probe59_startup']}, largest "
        f"difference {start['largest_face_difference']:.1e}). Per-step criterion "
        f"{rec['step_tol_over_U']:g} U, pressure_rtol {rec['pressure_rtol']:g}. "
        f"Commit {rec['commit']}. T2's rows are prompt 59's arms from the same field.",
        "",
        "| Arm | Scheme | dt | Courant | Steps | Outer per step: mean, median, min, max | Steps by count | Cap hits | Worst imbalance, last step | s per step | Processes |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for arm, data in rec["arms"].items():
        counts = ", ".join(f"{k}: {n}" for k, n in data["outer_counts"].items())
        lines.append(
            f"| {arm} | {data['scheme']} | {data['dt']:g} | {data['courant']:.2f} | "
            f"{data['steps']} | {data['outer_mean']:.2f}, {data['outer_median']:g}, "
            f"{data['outer_min']}, {data['outer_max']} | {counts} | {data['cap_hits']} | "
            f"{data['rows'][-1]['worst_imbalance']:.1e} | "
            f"{data['seconds_per_step_mean']:.2f} | {data['concurrent_processes']} |"
        )
    for name in ("A", "B"):
        t2 = load(PROBE59 / f"cylinder_{name}.json")
        if t2 is None:
            lines.append(f"| T2 {name} | | | | | not run | | | | | |")
            continue
        lines.append(
            f"| T2 {name} | implicit_euler | {t2['dt']:g} | {t2['courant']:.2f} | "
            f"{t2['steps']} | {t2['outer_mean']:.2f}, {t2['outer_median']:g}, "
            f"{min(r['outer'] for r in t2['rows'])}, {t2['outer_max']} | | "
            f"{sum(r['cap_hits'] for r in t2['rows'])} | "
            f"{t2['rows'][-1]['worst_imbalance']:.1e} | "
            f"{t2['seconds_per_step_mean']:.2f} | {t2['concurrent_processes']} |"
        )
    lines += ["", "### S2. The implicit Euler arm against the probe's arm A", ""]
    against = rec["arms"].get("A_ie", {}).get("against_probe59_arm_A")
    if against is None:
        return [*lines, "not run", ""]
    lines += [
        "| Probe commit | Steps compared | Outer equal | Wake v equal | Largest wake v difference |",
        "|---|---|---|---|---|",
        f"| {against['probe_commit']} | {against['steps_compared']} | "
        f"{against['outer_equal']} | {against['wake_v_equal']} | "
        f"{against['wake_v_largest_difference']:.1e} |",
        "",
    ]
    return lines


def hash_table() -> list[str]:
    """H1: every hash of record at this step's source against the earlier records."""
    lines = [
        "### H1. The hashes of record at the step 6a source",
        "",
        "hash62.py's solves from rest through the committed solver of the worktree "
        "named by the commit column: the face hash at this step's source (_67) "
        "against prompt 64's (_64) and prompt 62's (no suffix) records of the same "
        "case.",
        "",
        "| Case | Commit | Stop | Outer | Hash (first 16) | Equal to _64 | Equal to prompt 62's | Wall (s) | Processes |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for case in HASH_CASES:
        new = load(HASHES / f"hash_{case}_67.json")
        if new is None:
            lines.append(f"| {case} | | not run | | | | | | |")
            continue
        old64 = load(HASHES / f"hash_{case}_64.json")
        old62 = load(HASHES / f"hash_{case}.json")

        def same(old: dict[str, object] | None, new: dict[str, object] = new) -> str:
            """Whether a record's face hash is this step's, or that it has none."""
            return (
                "no record"
                if old is None
                else str(old["face_hash"] == new["face_hash"])
            )

        lines.append(
            f"| {case} | {new['commit']} | {new['stop']} | {new['outer']:,} | "
            f"{new['face_hash'][:16]} | {same(old64)} | {same(old62)} | "
            f"{new['wall_seconds']:.0f} | {new['concurrent_processes']} |"
        )
    return [*lines, ""]


def render() -> str:
    """The whole tables file."""
    head = [
        "# ADR-014 step 6a: tables",
        "",
        "Generated by `docs/reports/probe67/tables67.py` from the prompt 67 records; "
        "`--check` regenerates and compares. Do not edit by hand.",
        "",
    ]
    return "\n".join(head + order_tables() + smoke_tables() + hash_table())


def main() -> int:
    """Write the tables, or with --check compare them with the committed file."""
    parser = argparse.ArgumentParser(description="ADR-014 step 6a's tables")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    text = render()
    if args.check:
        committed = TABLES.read_text(encoding="utf-8").replace("\r\n", "\n")
        if committed != text:
            print("tables67.md differs from the records", file=sys.stderr)
            return 1
        print("tables67.md matches the records")
        return 0
    TABLES.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
