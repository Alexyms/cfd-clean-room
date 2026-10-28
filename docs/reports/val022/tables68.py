"""The tables of the VAL-022 report, generated from the prompt 68 records.

    python tables68.py            # writes docs/reports/val022/tables68.md
    python tables68.py --check    # regenerates and compares with the committed file

Reads docs/reports/val022/<preset>.json (scripts/benchmark.py's VAL-022 rows,
each with every step's lift and drag), the probes' records beside them
(rtol68_*.json, floor68.json, runout68.json, onset68.json), results/builder68/estimate_*.json
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
FLOOR_WINDOW = 5.0
RUNOUT_WINDOW = 20.0


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
        "Sohankar et al.: St 0.146, C_D 1.45 to 1.49; the band 0.139 to 0.153. "
        "The spectrum's resolution is one bin, about 9% to 10% of St here and wider "
        "than the band's 5%, so its agreement is a consistency check, not a "
        "discriminator: only the zero-crossing reading is scored.",
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


def decay_table() -> list[str]:
    """V7: Re 40's lift period by period: its length, amplitude and outer count."""
    lines = [
        "### V7. Re 40: the lift period by period",
        "",
        "Each whole period between upward zero crossings of C_L over the whole run: "
        "its length, half the range of C_L over it, and the mean outer iterations "
        "per step inside it (one when every step's first outer iteration already "
        "moves the field by less than the criterion).",
        "",
        "| Period from t | to t | Length | Amplitude | Mean outer per step |",
        "|---|---|---|---|---|",
    ]
    rec = load(HERE / "val022_re40.json")
    if rec is None:
        return [*lines, "not run", ""]
    t = np.array(rec["series"]["t"])
    c_l = np.array(rec["series"]["c_l"])
    outer = np.array(rec["series"]["outer"])
    up = np.nonzero((c_l[:-1] < 0.0) & (c_l[1:] >= 0.0))[0]
    for a, b in pairwise(up):
        span = c_l[a + 1 : b + 1]
        lines.append(
            f"| {t[a]:.2f} | {t[b]:.2f} | {t[b] - t[a]:.2f} | "
            f"{0.5 * (span.max() - span.min()):.3e} | {outer[a + 1 : b + 1].mean():.2f} |"
        )
    late = t > 100.0
    lines += [
        "",
        f"Steps after t = 100 taking one outer iteration: {np.mean(outer[late] == 1):.3f} "
        f"of {int(late.sum())}; the run ended at t = {t[-1]:g} after {t.size} steps.",
    ]
    return [*lines, ""]


def rtol_table() -> list[str]:
    """V8: the pressure_rtol probe (Alex, 2026-10-18): St at 1e-8 against 1e-4."""
    lines = [
        "### V8. Re 100 clustered: the Strouhal number at pressure_rtol 1e-8 and 1e-4",
        "",
        "Both arms continue the row's final checkpoint (step 4000, t = 200) by "
        "docs/reports/val022/rtol68.py, the case file with only pressure_rtol changed, "
        "side by side; the Strouhal number of the last ten periods of each. The "
        "amendment of ADR-013 decision 3 stands if 1e-8 and 1e-4 agree within 0.5%.",
        "",
        "| pressure_rtol | Steps | To t | St | Spectrum St | C_D | C_L amplitude | Outer per step: mean, max | Worst imbalance | s per step |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    found = {}
    for rtol in ("1e-08", "0.0001"):
        rec = load(HERE / f"rtol68_{rtol}.json")
        if rec is None:
            lines.append(f"| {rtol} | not run | | | | | | | | |")
            continue
        found[rtol] = rec
        lines.append(
            f"| {rec['pressure_rtol']:g} | {rec['steps']} | {rec['t_end']:g} | "
            f"{rec['strouhal']:.5f} | {rec['spectral_strouhal']:.5f} | "
            f"{rec['drag_coefficient']:.4f} | {rec['lift_amplitude']:.4f} | "
            f"{rec['outer_mean']:.2f}, {rec['outer_max']} | "
            f"{rec['worst_imbalance_max']:.1e} | {rec['seconds_per_step']:.2f} |"
        )
    row = load(HERE / "val022_re100.json")
    if len(found) == 2 and row is not None:
        tight, loose = found["1e-08"]["strouhal"], found["0.0001"]["strouhal"]
        lines += [
            "",
            f"Relative difference, 1e-8 against the 1e-4 arm: {(tight - loose) / loose:+.2e}; "
            f"against the row's St ({row['row']['accuracy']['strouhal']:.5f}): "
            f"{(tight - row['row']['accuracy']['strouhal']) / row['row']['accuracy']['strouhal']:+.2e}.",
        ]
    return [*lines, ""]


def floor_table() -> list[str]:
    """V9: the Re 40 floor probe (prompt 68b): the continuation at a step tolerance of 1e-6 U."""
    lines = [
        "### V9. Re 40 continued at a step tolerance of 1e-6 U",
        "",
        "From the row's final checkpoint (t = 200), docs/reports/val022/floor68.py; "
        "per window of the continuation, the largest |C_L|, the mean outer iterations "
        "per step and the mean C_D. Below: the envelope's rates (slope of ln(peak) "
        "in time; the original over the row's peaks from t = 63.70 to 101.30, the "
        "probe's from ten units in), predictions (e) and (f), and the row's measure "
        "on the continuation.",
        "",
        "| Window | Largest abs C_L | Mean outer per step | Mean C_D |",
        "|---|---|---|---|",
    ]
    rec = load(HERE / "floor68.json")
    if rec is None:
        return [*lines, "not run", ""]
    t = np.array(rec["series"]["t"])
    c_l = np.array(rec["series"]["c_l"])
    c_d = np.array(rec["series"]["c_d"])
    outer = np.array(rec["series"]["outer"])
    start = rec["from_t"]
    while start < t[-1] - 1e-9:
        inside = (t > start) & (t <= start + FLOOR_WINDOW + 1e-9)
        lines.append(
            f"| {start:g} to {start + FLOOR_WINDOW:g} | {np.max(np.abs(c_l[inside])):.3e} | "
            f"{np.mean(outer[inside]):.2f} | {np.mean(c_d[inside]):.6f} |"
        )
        start += FLOOR_WINDOW
    measure = rec["measure_on_continuation"]
    reading = (
        f"largest |C_L| over the last tenth {measure['largest_lift']:.3e}"
        if measure["measure"] == "quiet"
        else "ratios " + ", ".join(f"{r:.4f}" for r in measure["ratios"])
    )
    lines += [
        "",
        "| Reading | Value |",
        "|---|---|",
        f"| Steps, to t | {rec['steps']}, {rec['t_end']:g} (wall limit reached: {rec['stopped_by_wall']}) |",
        f"| Wall time (s) | {rec['wall_seconds']:.0f} |",
        f"| Original rate (per unit time) | {fmt(rec['original_rate'], '.4e')} |",
        f"| Probe rate (per unit time) | {fmt(rec['probe_rate'], '.4e')} |",
        f"| Ratio | {fmt(rec['rate_ratio'], '.3f')} |",
        f"| Last peak | {fmt(rec['last_peak'], '.3e')} |",
        f"| (e): ratio in [0.7, 1.3] and last peak below 1e-5 | {rec['prediction_e']} |",
        f"| Mean C_D, last ten units: probe, row | {rec['probe_drag']:.6f}, {rec['row_drag']:.6f} |",
        f"| (f): equal to four figures | {rec['prediction_f']} |",
        f"| The row's measure on the continuation ({measure['measure']}) | {reading}; passed {measure['passed']} |",
    ]
    return [*lines, ""]


def runout_table() -> list[str]:
    """V10: the Re 40 run-out (prompt 68c): the march to t = 600 at 1e-6 U and its measure."""
    lines = [
        "### V10. Re 40 run out to t = 600 at a step tolerance of 1e-6 U",
        "",
        "From the row's final checkpoint (t = 200), docs/reports/val022/runout68.py, "
        "the probe's settings (V9) to t = 600; per window of the run-out, the largest "
        "|C_L| (the envelope), the mean outer iterations per step and the mean C_D. "
        "Below: the rates, prediction (g) with its three parts, and the row's measure "
        "(`validation.cylinder.decays`, unchanged) on the whole series from t = 0, "
        "with the single-stage row's own reading beside it.",
        "",
        "| Window | Largest abs C_L | Mean outer per step | Mean C_D |",
        "|---|---|---|---|",
    ]
    rec = load(HERE / "runout68.json")
    if rec is None:
        return [*lines, "not run", ""]
    t = np.array(rec["series"]["t"])
    c_l = np.array(rec["series"]["c_l"])
    c_d = np.array(rec["series"]["c_d"])
    outer = np.array(rec["series"]["outer"])
    start = rec["from_t"]
    while start < t[-1] - 1e-9:
        inside = (t > start) & (t <= start + RUNOUT_WINDOW + 1e-9)
        lines.append(
            f"| {start:g} to {start + RUNOUT_WINDOW:g} | {np.max(np.abs(c_l[inside])):.3e} | "
            f"{np.mean(outer[inside]):.2f} | {np.mean(c_d[inside]):.6f} |"
        )
        start += RUNOUT_WINDOW

    def measure_text(measure: dict) -> str:
        if measure["measure"] == "quiet":
            reading = f"largest |C_L| over the last tenth {measure['largest_lift']:.3e}"
        else:
            reading = (
                "last amplitudes "
                + ", ".join(f"{a:.3e}" for a in measure["amplitudes"])
                + "; ratios "
                + ", ".join(f"{r:.4f}" for r in measure["ratios"])
            )
        return f"{measure['measure']}: {reading}; passed {measure['passed']}"

    same = rec["same_as_probe"]
    control = (
        "not compared"
        if same is None
        else ", ".join(f"{k} {same[k]}" for k in ("t", "outer", "c_l", "c_d"))
    )
    lines += [
        "",
        "| Reading | Value |",
        "|---|---|",
        f"| Steps, to t | {rec['steps']}, {rec['t_end']:g} (stopped by the budget: {rec['stopped_by_budget']}) |",
        f"| Wall time (s); of which to t = 240 | {rec['wall_seconds']:.0f}; {fmt(rec['probe_end_seconds'], '.0f')} |",
        f"| Estimate at t = 260 for the whole march (s) | {fmt(rec['estimate_total_seconds'], '.0f')} |",
        f"| First {same['steps'] if same else 0} steps equal to floor68.json's bit for bit ({control}) | "
        f"{all(same[k] for k in ('t', 'outer', 'c_l', 'c_d')) if same else 'not compared'} |",
        f"| Probe rate, t = 210 to 240 (per unit time) | {fmt(rec['probe_rate'], '.4e')} |",
        f"| Run-out rate, peaks from t = {rec['rate_from']:g} to the first below 1e-6 (per unit time) | {fmt(rec['run_rate'], '.4e')} |",
        f"| Ratio of the run-out rate to the probe rate | {fmt(rec['rate_ratio'], '.3f')} |",
        f"| Time of the first peak below 1e-6 | {fmt(rec['quiet_time'], '.2f')} |",
        f"| Last peak | {fmt(rec['last_peak'], '.3e')} |",
        f"| (g) rate within 30% of the probe's | {rec['prediction_g_rate']} |",
        f"| (g) first peak below 1e-6 inside {rec['quiet_time_window'][0]:g} to {rec['quiet_time_window'][1]:g} | {rec['prediction_g_quiet_time']} |",
        f"| (g) the measure passes on the series to {rec['t_end']:g} | {rec['prediction_g_measure']} |",
        f"| (g), all three | {rec['prediction_g']} |",
        f"| The measure on the whole series, t = 0 to {rec['t_end']:g} (1e-4 U, then 1e-6 U) | {measure_text(rec['measure_whole_series'])} |",
        f"| The measure on the single-stage row, t = 0 to 200 (1e-4 U) | {measure_text(rec['measure_row_alone'])} |",
    ]
    return [*lines, ""]


def onset_table() -> list[str]:
    """V11: the onset estimate (prompt 68c): the line through the Re 40 and Re 60 rates."""
    lines = [
        "### V11. An estimate of the onset of shedding from two growth rates",
        "",
        "Reported, not scored, except prediction (h). docs/reports/val022/onset68.py: "
        "the rate at Re 40 is the probe's late decay (V9), at Re 60 the envelope's "
        "growth between 1% and 10% of its saturated amplitude, where the Landau "
        "amplitude equation puts the rate within 1% of the linear one. The line "
        "through (40, rate) and (60, rate) crosses zero at the estimate.",
        "",
    ]
    rec = load(HERE / "onset68.json")
    if rec is None:
        return [*lines, "not run", ""]
    lines += [
        "| Re | Rate source | Window | Values | Rate per unit time | Largest residual of ln(amplitude) | Period |",
        "|---|---|---|---|---|---|---|",
        f"| 40 | floor68.json probe_rate | peaks from t = 210 to 240 | | {rec['rate_40']:.4e} | | {rec['period_40']:.2f} |",
    ]
    if rec["runout_rate_40"] is not None:
        lines.append(
            f"| 40 | runout68.json run_rate | peaks from t = 250 to the first below 1e-6 | | {rec['runout_rate_40']:.4e} | | |"
        )
    for name, reading in rec["re60_readings"].items():
        if reading["rate"] is None:
            lines.append(f"| 60 | {name} | not readable | {reading['values']} | | | |")
            continue
        lines.append(
            f"| 60 | {name} | t = {reading['from_t']:.1f} to {reading['to_t']:.1f}, "
            f"{reading['from_amplitude']:.3e} to {reading['to_amplitude']:.3e} | "
            f"{reading['values']} | {reading['rate']:.4e} | {reading['largest_residual']:.3f} | "
            f"{reading['period']:.2f} |"
        )
    lines += [
        "",
        f"Saturated amplitude at Re 60 (median over the last 50 units): {rec['saturated']:.4f}. "
        f"The Re 60 rate by the sign-change peaks of V9's definition over the primary window: "
        f"{fmt(rec['re60_sign_change_rate'], '.4e')}.",
        "",
        "| Reading | Estimate of Re_cr |",
        "|---|---|",
    ]
    for name, value in rec["estimate_sensitivity"].items():
        lines.append(f"| {name} (Re 40 rate from the probe) | {value:.1f} |")
    if rec["estimate_with_runout_rate"] is not None:
        lines.append(
            f"| primary window with the run-out's Re 40 rate | {rec['estimate_with_runout_rate']:.1f} |"
        )
    lines += [
        f"| Sohankar, Norberg and Davidson (Table V) | {rec['sohankar'][0]:g} +/- {rec['sohankar'][1]:g} |",
        f"| (h): the primary estimate inside {rec['predicted_band'][0]:g} to {rec['predicted_band'][1]:g} | {rec['prediction_h']} |",
    ]
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
    clustered = load(ESTIMATES / "estimate_clustered_0.05_0.0001_100.json")
    uniform = load(ESTIMATES / "estimate_uniform_0.05_0.0001_100.json")
    rows = {preset: load(HERE / f"{preset}.json") for preset in PRESETS}
    if clustered is not None and uniform is not None and None not in rows.values():
        total = 0.0
        for preset, row in rows.items():
            cost = uniform if preset.endswith("uniform") else clustered
            total += row["steps"] * cost["s_per_step_last100"] + cost["startup_s"]
        lines += [
            "",
            "The four rows' steps (V1) at the cost of their grid in the 0.05, 0.0001 "
            f"lines above, each with its start-up: {total:,.0f} s, {total / 3600:.1f} hours.",
        ]
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
        + decay_table()
        + rtol_table()
        + floor_table()
        + runout_table()
        + onset_table()
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
