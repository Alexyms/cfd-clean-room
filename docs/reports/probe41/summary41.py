"""Prompt 41: the report's tables from results/builder41/*.json, printed as Markdown.

Usage: python summary41.py
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results" / "builder41"
BASELINE_HASH = "c7d88d919e1141a59c6d8561875f370db6e548ba12d286a68d062b4e1cef0595"
BASELINE_METRIC = 0.0004107698619183384
BASELINE_OUTER = 1559


def load(name: str) -> dict | None:
    path = OUT / f"{name}.json"
    return json.loads(path.read_text()) if path.exists() else None


def fmt(x: float, digits: int = 3) -> str:
    return f"{x:.{digits - 1}e}"


def end_state(d: dict) -> str:
    if d["stop"] == "diverged":
        return "diverged"
    if d["stop"] == "max_simple_iter":
        res = np.array(d["residual"])
        tail = res[-100:]
        trend = "growing" if tail[-1] > tail[0] else "falling"
        return f"cap, {trend}"
    return d["stop"]


def last_window(d: dict, n: int) -> float:
    p = np.array(d["p_mean"])
    if p.size < n + 1:
        return float("nan")
    return float(np.mean(np.diff(p[-(n + 1) :])))


def drift_table() -> None:
    print("### Measurement 1: the drift case\n")
    print(
        "| Arm | rtol | Stop | velocity_step at | Outer | Mean p change per outer, last 100 (Pa) | last 10 | `\\|\\|b\\|\\|` end | Share beside open faces | Worst cell at end (kg/s per m) | Signed sum | Reversed, most at once | Held shut, most at once | CG per correction, median [max] | Cap hits | s |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    names = []
    for arm in ("A", "B", "C", "D"):
        names += [f"drift_{arm}_{r}" for r in ("1e-8", "1e-4", "1e-2")]
    names += ["drift_E_1e-8", "drift_E0_1e-8", "drift_A_1e-8_gradient"]
    names += [f"drift_F_{r}" for r in ("1e-8", "1e-4", "1e-2")]
    for name in names:
        d = load(name)
        if d is None:
            print(f"| {name} | (missing) |")
            continue
        label = d["arm"] + (
            " (A0, hood tangential zero gradient)" if name.endswith("gradient") else ""
        )
        print(
            f"| {label} | {d['pressure_rtol']:g} | {end_state(d)} | {d['velocity_step_outer']} | {d['outer']:,} | "
            f"{d['p_drift_last100']:+.3e} | {last_window(d, 10):+.1e} | {fmt(d['b_norm_end'])} | {d['b_share_end']:.3f} | "
            f"{fmt(d['worst_end'], 2)} | {d['signed_end']:+.1e} | {d['reversed_most']} | {d['closed_most']} | "
            f"{d['inner_median']:.0f} [{d['inner_max']}] | {d['cap_hits']} | {d['seconds']:.0f} |"
        )
    print()


def long_table() -> None:
    print(
        "### Measurement 1, continued: the pressure's movement past the stop (3,000 outer iterations, rtol 1e-8)\n"
    )
    marks = (200, 588, 1000, 2000, 3000)
    print(
        "| Arm | "
        + " | ".join(f"p change per outer at {m:,} (mean of 10)" for m in marks)
        + " | `\\|\\|b\\|\\|` at 3,000 | Residual at 3,000 | Worst cell at 3,000 |"
    )
    print("|---|" + "---|" * (len(marks) + 3))
    for arm in ("A", "B", "C", "D", "E", "E0", "F"):
        d = load(f"drift_{arm}_1e-8_long")
        if d is None:
            print(f"| {arm} | (missing) |")
            continue
        p = np.array(d["p_mean"])
        cells = []
        for m in marks:
            if p.size >= m:
                cells.append(f"{float(np.mean(np.diff(p[m - 10 : m]))):+.1e}")
            else:
                cells.append("-")
        print(
            f"| {arm} | "
            + " | ".join(cells)
            + f" | {fmt(d['b_norm_end'])} | {fmt(d['residual_end'])} | {fmt(d['worst_end'], 2)} |"
        )
    print()


def ladder_table() -> None:
    print("### Measurement 2: the ladder\n")
    print(
        "| Re | Arm | Outer | End | Residual: least, at end | Largest speed at end (m/s), cell | Reversed faces, most at once (returns 1 to 4, hood) | Held shut, most at once | Mean p change per outer, last 100 (Pa) | `\\|\\|b\\|\\|` end | s |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for rung in ("895", "8950"):
        for arm in ("A", "B", "C", "D", "E", "E0", "F"):
            d = load(f"ladder_{arm}_{rung}")
            if d is None:
                print(f"| {rung} | {arm} | (missing) |")
                continue
            print(
                f"| {int(rung):,} | {arm} | {d['outer']:,} | {end_state(d)} | {fmt(d['residual_min'])}, {fmt(d['residual_end'])} | "
                f"{d['max_speed_end']:.3g}, {tuple(d['at_end'])} | {d['reversed_most']} | {d['closed_most']} | "
                f"{d['p_drift_last100']:+.2e} | {fmt(d['b_norm_end'])} | {d['seconds']:.0f} |"
            )
    print()


def val001_table() -> None:
    print("### Measurement 3: VAL-001 80x40\n")
    print(
        "| Path | Outer | Stop | REQ-S02 metric | Equal to the baseline's 4.1077e-4 to three figures | Face hash (u then v) | Equals the baseline's | Mean p change per outer, last 100 (Pa) | last 10 | `\\|\\|b\\|\\|` end | Worst cell at end | Reversed faces, most | Held shut, most |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for arm in ("committed", "B", "C", "F"):
        d = load(f"val001_{arm}")
        if d is None:
            print(f"| {arm} | (missing) |")
            continue
        metric = d["metric"]["value"]
        three = f"{metric:.3e}" == f"{BASELINE_METRIC:.3e}"
        print(
            f"| {arm} | {d['outer']:,} ({(d['outer'] - BASELINE_OUTER) / BASELINE_OUTER:+.1%}) | {d['stop']} | {metric:.5e} | {three} | "
            f"{d['face_hash'][:16]}... | {d['face_hash'] == BASELINE_HASH} | {d['p_drift_last100']:+.2e} | {last_window(d, 10):+.1e} | "
            f"{fmt(d['b_norm_end'])} | {fmt(d['worst_end'], 2)} | {d['reversed_most']} | {d['closed_most']} |"
        )
    print()


def compare_table() -> None:
    print(
        "### Measurement 4: arm D's flow against arm B's at their stops (rtol 1e-8)\n"
    )
    print(
        "| Pair | max abs du (m/s), cell | max abs dv (m/s), cell | max abs dp after removing each mean (Pa), cell | Floor faces, max abs dv (m/s) | Scale: max abs u, v, p (B) |"
    )
    print("|---|---|---|---|---|---|")
    for pair in (
        ("drift_D_1e-8", "drift_B_1e-8"),
        ("drift_C_1e-8", "drift_B_1e-8"),
        ("drift_E_1e-8", "drift_B_1e-8"),
        ("drift_E_1e-8", "drift_D_1e-8"),
        ("drift_A_1e-8", "drift_B_1e-8"),
        ("drift_F_1e-8", "drift_B_1e-8"),
        ("drift_F_1e-8", "drift_A_1e-8"),
    ):
        d = load(f"compare_{pair[0]}_vs_{pair[1]}")
        if d is None:
            print(f"| {pair} | (missing) |")
            continue
        print(
            f"| {pair[0].split('_')[1]} vs {pair[1].split('_')[1]} | {d['u_c']['max_abs_diff']:.3e}, {tuple(d['u_c']['at'])} | "
            f"{d['v_c']['max_abs_diff']:.3e}, {tuple(d['v_c']['at'])} | {d['p_demeaned']['max_abs_diff']:.3e}, {tuple(d['p_demeaned']['at'])} | "
            f"{d['bottom_faces_max_abs_diff']:.3e} | {d['u_c']['scale']:.3f}, {d['v_c']['scale']:.3f}, {d['p_demeaned']['scale']:.3f} |"
        )
    print()


def grid_table() -> None:
    print(
        "### Measurement 3, continued: the committed path against arms B and F on three grids\n"
    )
    print(
        "| Grid | Path | Outer | REQ-S02 metric | Metric minus the committed path's | max abs du at cell centres vs committed (m/s), column | max abs v in the last column (m/s) | max abs (outlet face minus interior face) (m/s) |"
    )
    print("|---|---|---|---|---|---|---|---|")
    for grid in ("40x20", "80x40", "160x80"):
        suffix = "" if grid == "80x40" else f"_{grid}"
        ref = load(f"val001_committed{suffix}")
        ref_npz = OUT / f"val001_committed{suffix}.npz"
        for arm in ("committed", "B", "F"):
            d = load(f"val001_{arm}{suffix}")
            npz = OUT / f"val001_{arm}{suffix}.npz"
            if d is None or ref is None or not npz.exists() or not ref_npz.exists():
                print(f"| {grid} | {arm} | (missing) |")
                continue
            z, zr = np.load(npz), np.load(ref_npz)
            du = np.abs(z["u_c"] - zr["u_c"])
            _j, i = np.unravel_index(int(np.argmax(du)), du.shape)
            v_last = float(np.abs(z["v_faces"][:, -1]).max())
            gap = float(np.abs(z["u_faces"][:, -1] - z["u_faces"][:, -2]).max())
            print(
                f"| {grid} | {arm} | {d['outer']:,} | {d['metric']['value']:.5e} | {d['metric']['value'] - ref['metric']['value']:+.2e} | "
                f"{float(du.max()):.2e}, column {int(i)} of {du.shape[1]} | {v_last:.2e} | {gap:.2e} |"
            )
    print()


def main() -> None:
    which = sys.argv[1:] or ["drift", "long", "ladder", "val001", "grid", "compare"]
    for name in which:
        {
            "drift": drift_table,
            "long": long_table,
            "ladder": ladder_table,
            "val001": val001_table,
            "grid": grid_table,
            "compare": compare_table,
        }[name]()


if __name__ == "__main__":
    main()
