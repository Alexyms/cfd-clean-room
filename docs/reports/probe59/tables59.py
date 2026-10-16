"""The tables of ADR-014's probe section, generated from the prompt 59 records.

    python tables59.py            # writes docs/reports/probe59/tables59.md
    python tables59.py --check    # regenerates and compares with the committed file

Reads results/builder59/ (warm59.py, cylinder59.py, order59.py and scales59.py
records) and results/builder56/L0_320x120.json (the door-closed solve from
rest of ECR-002 step 8). A record that is missing prints as "not run". The
ADR copies the tables from the written file and points here; the prose
beside them states no number of its own.
"""

import argparse
import itertools
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RESULTS = ROOT / "results" / "builder59"
L0 = ROOT / "results" / "builder56" / "L0_320x120.json"
TABLES = HERE / "tables59.md"
STROUHAL = 0.146  # Sohankar et al. (1998), Table IV case 1, Re 100
SOHANKAR_STEPS = 6300  # time steps of 0.025 at Re 100 (the paper, section 4.1)
SOHANKAR_DT = 0.025
LONG_BAND = 200


def load(name: str) -> dict | None:
    """The record results/builder59/NAME.json as a mapping, or None when it was not written."""
    path = RESULTS / f"{name}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def room_table() -> list[str]:
    """T1: the product room re-solved to the door's states, and the warm-over-rest ratios."""
    lines = [
        "### T1. The product room re-solved to the door's states (320x120)",
        "",
        "| Arm | State solved | Start | Stop | Outer | Wall (s) | s per outer | Worst imbalance at end (kg/s per m) | Processes |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    if L0.exists():
        l0 = json.loads(L0.read_text(encoding="utf-8"))
        lines.append(
            f"| L0 (step 8) | door closed | rest | {l0['stop']} | {l0['outer']:,} | "
            f"{l0['seconds']:.0f} | {l0['seconds_per_outer']:.3f} | {l0['worst_end']:.2e} | "
            f"{l0['concurrent_processes']} |"
        )
    for arm, state in (
        ("rest", "door open"),
        ("warm", "door open"),
        ("back", "door closed"),
    ):
        rec = load(arm)
        if rec is None:
            lines.append(f"| {arm} | {state} | | not run | | | | | |")
            continue
        start = (
            "rest"
            if arm == "rest"
            else ("L0, converged" if arm == "warm" else "warm, converged")
        )
        lines.append(
            f"| {arm} | {state} | {start} | {rec['stop']} | {rec['outer']:,} | "
            f"{rec['wall_seconds']:.0f} | {rec['seconds_per_outer']:.3f} | "
            f"{rec['worst_imbalance_end']:.2e} | {rec['concurrent_processes']} |"
        )
    rest, warm, back = load("rest"), load("warm"), load("back")
    lines.append("")
    lines.append("| Ratio | Outer | Wall |")
    lines.append("|---|---|---|")
    if rest and warm:
        lines.append(
            f"| warm over rest (prediction (a): under one third) | "
            f"{warm['outer'] / rest['outer']:.3f} | {warm['wall_seconds'] / rest['wall_seconds']:.3f} |"
        )
    else:
        lines.append(
            "| warm over rest (prediction (a): under one third) | not run | not run |"
        )
    if rest and back:
        lines.append(
            f"| back over rest | {back['outer'] / rest['outer']:.3f} | "
            f"{back['wall_seconds'] / rest['wall_seconds']:.3f} |"
        )
    return lines


def cylinder_table() -> list[str]:
    """T2: outer iterations per implicit Euler step for arms A to F, and the start-up line."""
    lines = [
        "### T2. The cylinder at Re 100: outer iterations per implicit Euler step, from the start-up field",
        "",
        "| Arm | dt | Courant | pressure_rtol | Step tol (U) | Outer per step: mean, median, max | Mean outer to 1e-3, 1e-4, 1e-5, 1e-6 | s per step | s per outer | Imbalance (last step) | Processes |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for arm in "ABCDEF":
        rec = load(f"cylinder_{arm}")
        if rec is None:
            lines.append(f"| {arm} | | | | | not run | | | | | |")
            continue
        at = rec.get("outer_at_mean", {})
        counts = ", ".join(
            f"{at[key]:.1f}" if key in at else "-"
            for key in ("0.001", "0.0001", "1e-05", "1e-06")
        )
        lines.append(
            f"| {arm} | {rec['dt']} | {rec['courant']:.2f} | {rec['pressure_rtol']:g} | "
            f"{rec['step_tol_over_U']:g} | {rec['outer_mean']:.1f}, {rec['outer_median']:g}, "
            f"{rec['outer_max']} | {counts} | {rec['seconds_per_step_mean']:.2f} | "
            f"{rec['seconds_per_outer']:.3f} | {rec['rows'][-1]['worst_imbalance']:.1e} | "
            f"{rec['concurrent_processes']} |"
        )
    arm_a = load("cylinder_A")
    startup = None if arm_a is None else load(f"cylinder_startup_dx{arm_a['dx']:g}")
    lines.append("")
    if startup is None:
        lines.append("Start-up record: not kept.")
    else:
        lines.append(
            f"Start-up, rerun after the arms (the field the arms began from was overwritten by "
            f"a smoke run and is not kept; the steady loop is deterministic, so the rerun is the "
            f"same field): {startup['outer']} steady outer iterations from rest on "
            f"{startup['grid'][0]}x{startup['grid'][1]} "
            f"({startup['grid'][0] * startup['grid'][1]:,} cells), "
            f"{startup['seconds_per_outer']:.3f} s per outer with "
            f"{startup['concurrent_processes']} processes."
        )
    return lines


LONG_ARMS = ("B_long", "D_long")


def long_table() -> list[str]:
    """T3: the long marches in bands of LONG_BAND steps, with the wake sign changes."""
    lines = [
        "### T3. The long marches from the perturbed start-up: the cost as shedding develops",
        "",
    ]
    period = 1.0 / STROUHAL
    for arm in LONG_ARMS:
        rec = load(f"cylinder_{arm}")
        if rec is None:
            lines.append(f"Arm {arm}: not run, or not completed.")
            lines.append("")
            continue
        lines.append(
            f"Arm {arm}: dt {rec['dt']}, step tolerance {rec['step_tol_over_U']:g} U, "
            f"{rec['steps']} steps ({rec['steps'] * rec['dt']:.0f} time units, "
            f"{rec['steps'] * rec['dt'] / period:.1f} periods at St {STROUHAL}); wake sign "
            f"changes over the march: {rec['wake_sign_changes']}; "
            f"{rec['concurrent_processes']} processes."
        )
        lines.append("")
        lines.append(
            "| Steps | Time units | Mean outer per step | Max outer | s per step | Largest wake v | Sign changes in band | Time units between sign changes (twice is a period) |"
        )
        lines.append("|---|---|---|---|---|---|---|---|")
        rows = rec["rows"]
        for start in range(0, len(rows), LONG_BAND):
            band = rows[start : start + LONG_BAND]
            outer = [r["outer"] for r in band]
            wake = [r["wake_v"] for r in band]
            changes = [
                b["t"]
                for a, b in itertools.pairwise(band)
                if (a["wake_v"] < 0) != (b["wake_v"] < 0)
            ]
            spacing = (
                f"{(changes[-1] - changes[0]) / (len(changes) - 1):.2f}"
                if len(changes) > 1
                else "-"
            )
            lines.append(
                f"| {band[0]['step']} to {band[-1]['step']} | {band[0]['t']:.1f} to "
                f"{band[-1]['t']:.1f} | {sum(outer) / len(outer):.1f} | {max(outer)} | "
                f"{sum(r['wall_seconds'] for r in band) / len(band):.2f} | "
                f"{max(abs(w) for w in wake):.4f} | {len(changes)} | {spacing} |"
            )
        lines.append("")
    return lines


def showcase_table() -> list[str]:
    """T4: hours for one run of the paper's length from each arm's seconds per step."""
    lines = [
        "### T4. The showcase's cost from T2 and T3: one run at Re 100 to Sohankar's length",
        "",
        f"Period 1 / St = {1.0 / STROUHAL:.2f} time units at St {STROUHAL}; the paper's run is "
        f"{SOHANKAR_STEPS} steps of {SOHANKAR_DT} = {SOHANKAR_STEPS * SOHANKAR_DT:.1f} time units, "
        f"{SOHANKAR_STEPS * SOHANKAR_DT * STROUHAL:.1f} periods. Hours = steps x seconds per step.",
        "",
        "| Arm | dt | Steps per period | Steps for the paper's length | s per step (measured) | Hours, one run |",
        "|---|---|---|---|---|---|",
    ]
    for arm in ("A", "B", "C", "D", "E", "F", *LONG_ARMS):
        rec = load(f"cylinder_{arm}")
        if rec is None:
            continue
        per_period = (1.0 / STROUHAL) / rec["dt"]
        steps = SOHANKAR_STEPS * SOHANKAR_DT / rec["dt"]
        if arm in LONG_ARMS:
            band = rec["rows"][-LONG_BAND:]
            seconds = sum(r["wall_seconds"] for r in band) / len(band)
            label = f"{arm} (last band)"
        else:
            seconds = rec["seconds_per_step_mean"]
            label = arm
        lines.append(
            f"| {label} | {rec['dt']} | {per_period:.0f} | {steps:.0f} | {seconds:.2f} | "
            f"{steps * seconds / 3600.0:.2f} |"
        )
    return lines


def order_table() -> list[str]:
    """T5: the observed orders of order59.py's cases."""
    data = load("order59")
    lines = [
        "### T5. The observed order of the transport scheme under refinement (VAL-008)",
        "",
    ]
    if data is None:
        lines.append("not run")
        return lines
    lines.append(
        "| Case | Scheme | Courant rule | Grids (dx, m) | Relative L2 error per grid | Peak retained per grid | Observed order between grids |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for r in data:
        grids = ", ".join(f"{row['dx']:g}" for row in r["rows"])
        errors = ", ".join(f"{row['l2']:.3e}" for row in r["rows"])
        peaks = ", ".join(f"{row['peak']:.3f}" for row in r["rows"])
        orders = ", ".join(f"{o:.2f}" for o in r["orders"])
        lines.append(
            f"| {r['case']} | {r['scheme']} | {r['rule']} | {grids} | {errors} | {peaks} | {orders} |"
        )
    return lines


def scales_table() -> list[str]:
    """T6: the room's scales and the transport march's cost from scales59.py."""
    s = load("scales59")
    lines = [
        "### T6. The product room's scales and the transport march's cost (320x120, L0 faces)",
        "",
    ]
    if s is None:
        lines.append("not run")
        return lines
    lines.append("| Quantity | Value |")
    lines.append("|---|---|")
    rows = (
        ("Free area of the slice (m^2)", s["free_area_m2"]),
        ("Supply flow (m^2/s per m depth)", s["supply_flow_m2_per_s"]),
        ("Hood flow (m^2/s)", s["hood_flow_m2_per_s"]),
        ("Return flow, four returns (m^2/s)", s["return_flow_m2_per_s"]),
        ("Return face velocity (m/s)", s["return_face_velocity_m_per_s"]),
        ("Air change rate N (1/s)", s["air_change_rate_per_s"]),
        ("Air changes per hour", s["air_changes_per_hour"]),
        ("Turnover 1 / N (s)", s["turnover_s"]),
        (
            "Well-mixed 100:1 recovery ln(100) / N (s)",
            s["recovery_100_to_1_well_mixed_s"],
        ),
        ("Room length over supply speed (s)", s["length_over_supply_speed_s"]),
        ("Room height over supply speed (s)", s["height_over_supply_speed_s"]),
        ("Transport cfl_number", s["cfl_number"]),
        (
            "Wall seconds per simulated second, five classes",
            s["wall_seconds_per_simulated_second_all_classes"],
        ),
    )
    for label, value in rows:
        lines.append(f"| {label} | {value:.4g} |")
    lines.append("")
    lines.append(
        "| Class (um) | Settling velocity (m/s) | Height over settling (s) | Stable dt at cfl (s) | Steps per simulated s | Wall s per step | Wall s per simulated s |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for c in s["classes"]:
        lines.append(
            f"| {c['size_m'] * 1e6:g} | {c['settling_velocity_m_per_s']:.3e} | "
            f"{c['settling_time_over_height_s']:.4g} | {c['stable_dt_s']:.4g} | "
            f"{c['steps_per_simulated_second']:.4g} | {c['seconds_per_step']:.4g} | "
            f"{c['wall_seconds_per_simulated_second']:.4g} |"
        )
    return lines


CLEANOUT_ARMS = ("0.1", "0.4")
CLEANOUT_MARKS = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0)  # in air changes 1 / N


def cleanout_table() -> list[str]:
    """T7: the clean-out march at Courant 0.1 and 0.4 (prompt 59b), the time error itself."""
    lines = [
        "### T7. The clean-out at Courant 0.1 and 0.4 (160x60, L0 flow, Sc_t 0.7): the time error",
        "",
    ]
    arms = {c: load(f"cleanout_cfl{c}") for c in CLEANOUT_ARMS}
    if any(rec is None for rec in arms.values()):
        lines.append("not run")
        return lines
    first = arms[CLEANOUT_ARMS[0]]
    lines.append(
        f"C0 = {first['c0']:g} in every non-SOLID cell at t = 0, the supply clean, the step the "
        f"smaller stable_dt of the two classes ({first['sizes_m'][0] * 1e6:g} and "
        f"{first['sizes_m'][1] * 1e6:g} um), run until every sensor of both classes is below "
        f"1e-3 C0; N = {first['air_change_rate_per_s']:.4f} per second (1 / N = "
        f"{1.0 / first['air_change_rate_per_s']:.2f} s)."
    )
    lines.append("")
    lines.append(
        "| Arm | dt (s) | Steps | Simulated end (s) | Stopped by | Wall (s) | Wall s per simulated s | Budget relative residual | Processes |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for c, rec in arms.items():
        budget = ", ".join(f"{v:.1e}" for v in rec["budget_relative"].values())
        lines.append(
            f"| {c} | {rec['dt']:.4e} | {rec['steps']:,} | {rec['t_end']:.2f} | {rec['stopped_by']} | "
            f"{rec['wall_seconds']:.0f} | {rec['wall_seconds_per_simulated_second']:.2f} | "
            f"{budget} | {rec['concurrent_processes']} |"
        )
    lines.append("")
    lines.append(
        "| Class (um) | Sensor | Level | First time below and staying below, Courant 0.1 (s) | Courant 0.4 (s) | 0.4 against 0.1 |"
    )
    lines.append("|---|---|---|---|---|---|")
    worst = 0.0
    a, b = arms[CLEANOUT_ARMS[0]], arms[CLEANOUT_ARMS[1]]
    for k, size in zip(a["classes"], a["sizes_m"], strict=True):
        for sensor, levels in a["recovery"][str(k)].items():
            for level, t_a in levels.items():
                t_b = b["recovery"][str(k)][sensor][level]
                if t_a is None or t_b is None:
                    diff = "not reached"
                else:
                    rel = (t_b - t_a) / t_a if t_a > 0 else 0.0
                    worst = max(worst, abs(rel))
                    diff = f"{rel * 100:+.2f}%"
                fa = "not reached" if t_a is None else f"{t_a:.2f}"
                fb = "not reached" if t_b is None else f"{t_b:.2f}"
                lines.append(
                    f"| {size * 1e6:g} | {sensor} | {float(level):g} | {fa} | {fb} | {diff} |"
                )
        for level, t_a in a["mean_recovery"][str(k)].items():
            t_b = b["mean_recovery"][str(k)][level]
            fa = "not reached" if t_a is None else f"{t_a:.2f}"
            fb = "not reached" if t_b is None else f"{t_b:.2f}"
            diff = (
                "-"
                if t_a is None or t_b is None
                else f"{(t_b - t_a) / t_a * 100:+.2f}%"
            )
            lines.append(
                f"| {size * 1e6:g} | free-area mean | {float(level):g} | {fa} | {fb} | {diff} |"
            )
    lines.append("")
    lines.append(
        f"Largest relative difference of a sensor's recovery time, 0.4 against 0.1: "
        f"{worst * 100:.2f}% (prediction (c): under 2%)."
    )
    lines.append("")
    lines.append(
        "| Air changes N t | t (s) | Well-mixed exp(-N t) | Free-area mean, 0.5 um, Courant 0.1 | 0.4 | Free-area mean, 5 um, Courant 0.1 | 0.4 |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    rate = a["air_change_rate_per_s"]
    import math

    for mark in CLEANOUT_MARKS:
        t = mark / rate
        cells = []
        for rec in (a, b):
            times = rec["times"]
            idx = min(range(len(times)), key=lambda i: abs(times[i] - t))
            cells.append(
                (
                    rec["means"][str(a["classes"][0])][idx],
                    rec["means"][str(a["classes"][1])][idx],
                    times[idx],
                )
            )
        if cells[0][2] > a["t_end"] + 1e-9 or t > a["t_end"] + 1e-9:
            break
        lines.append(
            f"| {mark:g} | {t:.1f} | {math.exp(-mark):.3e} | {cells[0][0]:.3e} | {cells[1][0]:.3e} | "
            f"{cells[0][1]:.3e} | {cells[1][1]:.3e} |"
        )
    return lines


def render() -> str:
    """The whole tables file as text."""
    parts = [
        "# Prompt 59 probe tables",
        "",
        "Generated by `docs/reports/probe59/tables59.py` from `results/builder59/` and "
        "`results/builder56/L0_320x120.json`; `--check` regenerates and compares. Wall times "
        "were taken with the stated number of processes sharing the machine and are not like "
        "for like across tables.",
        "",
    ]
    for block in (
        room_table,
        cylinder_table,
        long_table,
        showcase_table,
        order_table,
        scales_table,
        cleanout_table,
    ):
        parts.extend(block())
        parts.append("")
    return "\n".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description="ADR-014's probe tables")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    text = render()
    if args.check:
        committed = TABLES.read_text(encoding="utf-8").replace("\r\n", "\n")
        if committed != text:
            print("tables59.md differs from the records", file=sys.stderr)
            return 1
        print("tables59.md matches the records")
        return 0
    TABLES.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
