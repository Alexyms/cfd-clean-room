"""The tables of the VAL-022 report, generated from the prompt 68 records.

    python tables68.py            # writes docs/reports/val022/tables68.md
    python tables68.py --check    # regenerates and compares with the committed file

Reads docs/reports/val022/<preset>.json (scripts/benchmark.py's VAL-022 rows,
each with every step's lift and drag), results/builder68/estimate_*.json
(the cost estimate) and the hash records results/builder62/hash_*_68.json
(hash62.py at this step's source) against prompt 67's (_67). A record that is
missing prints as "not run". The report, val022.md, copies the tables and
states no number of its own.
"""

import argparse
import json
import sys
from itertools import pairwise
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
TABLES = HERE / "tables68.md"
ESTIMATES = ROOT / "results" / "builder68"
HASHES = ROOT / "results" / "builder62"
ROWS = ROOT / "benchmarks" / "results.jsonl"
PRESETS = ("val022_re40", "val022_re60", "val022_re100", "val022_re100_uniform")
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
HISTORY_PERIODS = 20
WINDOW = 50.0


def load(path: Path) -> dict | None:
    """The JSON record at PATH as a mapping, or None when it was not written."""
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def crossings(t: np.ndarray, c_l: np.ndarray) -> np.ndarray:
    """The interpolated upward zero crossings of the lift, as forces.strouhal places them."""
    k = np.nonzero((c_l[:-1] < 0.0) & (c_l[1:] >= 0.0))[0]
    return t[k] - c_l[k] * (t[k + 1] - t[k]) / (c_l[k + 1] - c_l[k])


def fmt(value: object, spec: str) -> str:
    """A number by spec, or a dash for None and NaN."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "-"
    return format(value, spec)


def harness_rows() -> dict[str, dict]:
    """benchmarks/results.jsonl's rows by run id."""
    if not ROWS.exists():
        return {}
    rows = (
        json.loads(line)
        for line in ROWS.read_text(encoding="utf-8").splitlines()
        if line
    )
    return {row["run_id"]: row for row in rows}


def rows_table() -> list[str]:
    """V1: each row against its criterion, the cap hits from the harness row."""
    stored = harness_rows()
    lines = [
        "### V1. The rows against their criteria",
        "",
        "| Preset | Re | Grid | Steps | Outer per step: mean, max | Cap hits | Criterion | Reading | Passed | Wall (s) | Processes | Commit |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for preset in PRESETS:
        rec = load(HERE / f"{preset}.json")
        if rec is None:
            lines.append(f"| {preset} | | | | | | | not run | | | | |")
            continue
        acc = rec["row"]["accuracy"]
        grid = rec["row"]["grid"]
        lines.append(
            f"| {preset} | {acc['reynolds']:g} | {grid['nx']}x{grid['ny']} | "
            f"{rec['steps']} | {rec['outer_mean']:.2f}, {rec['outer_max']} | "
            f"{stored.get(rec['row']['run_id'], {}).get('outcome', {}).get('pressure_cap_hits', '-')}"
            f" | {acc['metric']} | {fmt(acc['value'], '.5f')} | {acc['passed']} | "
            f"{rec['row']['time']['wall_seconds']:.0f} | {rec['concurrent_processes']} | "
            f"{rec['row']['git_commit'][:7]}{' (dirty)' if rec['row']['git_dirty'] else ''} |"
        )
    return [*lines, ""]


def strouhal_table() -> list[str]:
    """V2: Re 100 on both grids: St, the spectrum's cross-check, C_D, the lift's amplitude."""
    lines = [
        "### V2. Re 100: the Strouhal number, the drag coefficient, the lift",
        "",
        "St from the last ten periods' interpolated zero crossings, beside the "
        "spectrum's peak over the same window and its resolution; C_D the mean "
        "over that window; the lift amplitude half the range of C_L over it. "
        "Sohankar et al.: St 0.146, C_D 1.45 to 1.49; the band 0.139 to 0.153.",
        "",
        "| Preset | Grid | St | Spectrum St | Resolution | Agree | C_D | C_L amplitude | In band |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for preset in ("val022_re100", "val022_re100_uniform"):
        rec = load(HERE / f"{preset}.json")
        if rec is None:
            lines.append(f"| {preset} | | not run | | | | | | |")
            continue
        acc = rec["row"]["accuracy"]
        grid = rec["row"]["grid"]
        t = np.array(rec["series"]["t"])
        c_l = np.array(rec["series"]["c_l"])
        found = crossings(t, c_l)
        amplitude = None
        if found.size >= 11:
            window = t >= found[-11]
            amplitude = 0.5 * float(c_l[window].max() - c_l[window].min())
        band = acc["strouhal"] is not None and 0.139 <= acc["strouhal"] <= 0.153
        lines.append(
            f"| {preset} | {grid['nx']}x{grid['ny']} | {fmt(acc['strouhal'], '.5f')} | "
            f"{fmt(acc['spectral_strouhal'], '.5f')} | {fmt(acc['spectral_resolution'], '.5f')} | "
            f"{acc['spectra_agree']} | {fmt(acc['drag_coefficient'], '.4f')} | "
            f"{fmt(amplitude, '.4f')} | {band} |"
        )
        if acc["strouhal"] is None:
            lines.append(f"| | refusal: {acc['strouhal_refusal']} | | | | | | | |")
    return [*lines, ""]


def amplitude_table() -> list[str]:
    """V3: the onset bracket's measures: the last six amplitudes and their ratios."""
    lines = [
        "### V3. Re 40 and Re 60: the lift's last amplitudes",
        "",
        "Half the range of C_L over each whole period between upward zero crossings, "
        "the last six; Re 40 passes when each is below the one before, Re 60 when "
        "each is at least 0.99 of the one before.",
        "",
        "| Preset | Measure | Amplitudes, oldest first | Ratios | Passed |",
        "|---|---|---|---|---|",
    ]
    for preset in ("val022_re40", "val022_re60"):
        rec = load(HERE / f"{preset}.json")
        if rec is None:
            lines.append(f"| {preset} | not run | | | |")
            continue
        measure = rec["row"]["accuracy"]["amplitude_measure"]
        amps = ", ".join(f"{a:.3e}" for a in measure.get("amplitudes", []))
        ratios = ", ".join(f"{r:.4f}" for r in measure.get("ratios", []))
        if measure["measure"] == "quiet":
            amps = f"largest |C_L| over the last tenth {measure['largest_lift']:.3e}"
        lines.append(
            f"| {preset} | {measure['measure']} | {amps} | {ratios} | {measure['passed']} |"
        )
    return [*lines, ""]


def history_table() -> list[str]:
    """V4: the period history of the Re 100 runs, the last twenty periods."""
    lines = [
        "### V4. Re 100: the period history",
        "",
        f"The time between successive upward zero crossings of C_L, the last "
        f"{HISTORY_PERIODS} periods, with the Strouhal number each period alone gives.",
        "",
        "| Preset | Period ends at t | Period | St of that period |",
        "|---|---|---|---|",
    ]
    for preset in ("val022_re100", "val022_re100_uniform"):
        rec = load(HERE / f"{preset}.json")
        if rec is None:
            lines.append(f"| {preset} | not run | | |")
            continue
        t = np.array(rec["series"]["t"])
        c_l = np.array(rec["series"]["c_l"])
        found = crossings(t, c_l)
        pairs = list(pairwise(found))[-HISTORY_PERIODS:]
        for a, b in pairs:
            lines.append(f"| {preset} | {b:.3f} | {b - a:.4f} | {1.0 / (b - a):.5f} |")
    return [*lines, ""]


def window_table() -> list[str]:
    """V6: every row by windows of WINDOW time units: the lift, the drag, the outer count."""
    lines = [
        "### V6. Each row by time window",
        "",
        f"Over each {WINDOW:g} units of the run: the largest |C_L|, the mean C_D and the "
        "mean outer iterations per step (a step whose first outer iteration already "
        "moves the field by less than the criterion takes one).",
        "",
        "| Preset | Window | Largest abs C_L | Mean C_D | Mean outer per step |",
        "|---|---|---|---|---|",
    ]
    for preset in PRESETS:
        rec = load(HERE / f"{preset}.json")
        if rec is None:
            lines.append(f"| {preset} | not run | | | |")
            continue
        t = np.array(rec["series"]["t"])
        c_l = np.array(rec["series"]["c_l"])
        c_d = np.array(rec["series"]["c_d"])
        outer = np.array(rec["series"]["outer"])
        start = 0.0
        while start < t[-1]:
            inside = (t > start) & (t <= start + WINDOW)
            lines.append(
                f"| {preset} | {start:g} to {start + WINDOW:g} | "
                f"{np.max(np.abs(c_l[inside])):.3e} | {np.mean(c_d[inside]):.5f} | "
                f"{np.mean(outer[inside]):.2f} |"
            )
            start += WINDOW
    return [*lines, ""]


def cost_table() -> list[str]:
    """V5: the cost estimate from 300 steps of each grid (prompt 68's budget rule)."""
    lines = [
        "### V5. The cost estimate behind the budget",
        "",
        "300 steps of Re 100 from the perturbed start-up, four processes sharing the "
        "machine (results/builder68/estimate68.py).",
        "",
        "| Grid | Cells | dt | pressure_rtol | Start-up (s) | Outer per step: mean, max | s per step, last 100 |",
        "|---|---|---|---|---|---|---|",
    ]
    for name in (
        "estimate_clustered_0.05_0.0001_100.json",
        "estimate_clustered_0.05_1e-08_100.json",
        "estimate_clustered_0.025_0.0001_100.json",
        "estimate_uniform_0.05_0.0001_100.json",
    ):
        rec = load(ESTIMATES / name)
        if rec is None:
            lines.append(f"| {name} | | | | | not run | |")
            continue
        lines.append(
            f"| {rec['grid']} | {rec['cells']:,} | {rec['dt']:g} | {rec['rtol']:g} | "
            f"{rec['startup_s']:.0f} | {rec['outer_mean']:.2f}, {rec['outer_max']} | "
            f"{rec['s_per_step_last100']:.2f} |"
        )
    return [*lines, ""]


def hash_table() -> list[str]:
    """H1: every hash of record at this step's source against prompt 67's."""
    lines = [
        "### H1. The hashes of record at the step 6b source",
        "",
        "hash62.py's solves from rest through the committed solver of the worktree "
        "the commit column names, against prompt 67's records (_67).",
        "",
        "| Case | Commit | Stop | Outer | Hash (first 16) | Equal to _67 | Wall (s) | Processes |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for case in HASH_CASES:
        new = load(HASHES / f"hash_{case}_68.json")
        if new is None:
            lines.append(f"| {case} | | not run | | | | | |")
            continue
        old = load(HASHES / f"hash_{case}_67.json")
        same = "no record" if old is None else str(old["face_hash"] == new["face_hash"])
        lines.append(
            f"| {case} | {new['commit']} | {new['stop']} | {new['outer']:,} | "
            f"{new['face_hash'][:16]} | {same} | {new['wall_seconds']:.0f} | "
            f"{new['concurrent_processes']} |"
        )
    return [*lines, ""]


def render() -> str:
    """The whole tables file."""
    head = [
        "# VAL-022: tables",
        "",
        "Generated by `docs/reports/val022/tables68.py` from the prompt 68 records; "
        "`--check` regenerates and compares. Do not edit by hand.",
        "",
    ]
    return "\n".join(
        head
        + rows_table()
        + strouhal_table()
        + amplitude_table()
        + history_table()
        + window_table()
        + cost_table()
        + hash_table()
    )


def main() -> int:
    """Write the tables, or with --check compare them with the committed file."""
    parser = argparse.ArgumentParser(description="VAL-022's tables")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    text = render()
    if args.check:
        committed = TABLES.read_text(encoding="utf-8").replace("\r\n", "\n")
        if committed != text:
            print("tables68.md differs from the records", file=sys.stderr)
            return 1
        print("tables68.md matches the records")
        return 0
    TABLES.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
