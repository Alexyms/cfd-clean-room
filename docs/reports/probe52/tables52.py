"""The report's tables from the prompt 52 records.

Usage:
    python tables52.py precursor    item 2: the layer along the channel, the two lengths
    python tables52.py matrix       item 4: every flow run, classified by step 5's rule
    python tables52.py walls        item 4: y+ on the three walls, the inflow check,
                                    the outlet's drift reading, per converged run
    python tables52.py reattachment item 5: x_r on every converged run, the data, the
                                    published k-epsilon runs; the resolution study
    python tables52.py profiles     item 5: each converged run at each station against
                                    the data (largest, RMS); the inflow profile
    python tables52.py coefficients item 5: Cf and Cp along the bottom wall against the
                                    data, per converged run
    python tables52.py schemes      item 5: upwind against umist on the medium grid
    python tables52.py predictions  section 6: (a) to (f) scored from the records
    python tables52.py timing       the run times
    python tables52.py all          everything above

Reads results/builder52/NAME.json, NAME.npz and NAME_pmean.json
(backstep52.py) and precursor_precursor.json (precursor52.py), prints
markdown tables, and keeps every number it printed in
results/builder52/tables_MODE.json. The classification is tables46's
(imported): diverged, converged, growing, bounded with step 0's
sub-classes, or a positivity error, its growing threshold scaled to the
case's speed. The comparison numbers are
validation.backstep.evaluate's, recomputed here from each record's final
faces, pressure and k, so the tables and the harness row read the same code.
"""

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "docs" / "reports" / "probe46"))

import tables46  # noqa: E402
from tables46 import classify, fmt, print_table  # noqa: E402

from src.config import SimConfig  # noqa: E402
from src.mesh import Mesh  # noqa: E402
from validation import backstep  # noqa: E402

OUT = ROOT / "results" / "builder52"
tables46.OUT = OUT
# tables46's growing threshold is 5 m/s, ten times a room's supply speed; the
# step runs at U_ref 44.2 m/s, so the same ratio is ten times U_ref (the
# runner's divergence guard).
tables46.GROWING_SPEED = 10.0 * backstep.U_REF
GRIDS = ("410x40", "656x60", "984x90")
GRID_LABEL = {"410x40": "coarse", "656x60": "medium", "984x90": "fine"}
REFINEMENT = 1.5
SCHEMES = ("upwind", "umist")
VARIANTS = ("standard", "rng")
NAME = re.compile(
    r"^(?P<variant>standard|rng)_(?P<grid>\d+x\d+)_(?P<scheme>upwind|umist)$"
)
CONVERGED = "error_estimate_and_continuity"
AT_CAP_COMPARED = "bounded"
# Prediction thresholds (section 3 of the report).
B_RANGE = (5.0, 5.9)
C_TOL_H = 0.2
D_TOL_H = 0.2
D_RATIO = 0.5
E_TOL = 0.1
E_STATIONS = (1.0, 4.0)
# Prediction (e)'s "outside the recirculation": heights above the step's top.
E_ABOVE_H = 1.0


def keep(mode: str, data: object) -> None:
    """The printed numbers, as JSON beside the records."""
    (OUT / f"tables_{mode}.json").write_text(json.dumps(data, indent=1, default=float))


def records() -> dict[str, dict]:
    """Every flow record of item 4, keyed by name, in grid, variant, scheme order."""
    found = {}
    for path in sorted(OUT.glob("*.json")):
        m = NAME.match(path.stem)
        if not m:
            continue
        rec = json.loads(path.read_text())
        if "stop" not in rec:
            continue
        rec["_grid"], rec["_variant"], rec["_scheme"] = (
            m["grid"],
            m["variant"],
            m["scheme"],
        )
        cls, sub = classify(rec)
        rec["_class"] = cls + (f" ({sub})" if sub else "")
        found[path.stem] = rec
    return dict(
        sorted(
            found.items(),
            key=lambda kv: (
                GRIDS.index(kv[1]["_grid"]),
                VARIANTS.index(kv[1]["_variant"]),
                SCHEMES.index(kv[1]["_scheme"]),
            ),
        )
    )


def compared(rec: dict) -> bool:
    """Whether a record's fields enter the comparison: converged, or bounded at the cap."""
    return rec["stop"] == CONVERGED or rec["_class"].startswith(AT_CAP_COMPARED)


def evaluation(name: str, rec: dict) -> dict:
    """validation.backstep.evaluate on a record's final fields, cached beside it."""
    cache = OUT / f"eval_{name}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    arrays = np.load(OUT / f"{name}.npz")
    config = SimConfig.from_dict(rec["raw"])
    mesh = Mesh(config)
    report = backstep.evaluate(
        mesh, config, arrays["u_faces"], arrays["p"], arrays["k"]
    )
    cache.write_text(json.dumps(report, indent=1))
    return report


def compared_records() -> dict[str, tuple[dict, dict]]:
    """Each compared record with its evaluation."""
    out = {}
    for name, rec in records().items():
        if compared(rec) and rec.get("faces_kept"):
            out[name] = (rec, evaluation(name, rec))
    return out


def readings_cell(rec: dict) -> str:
    r = rec.get("readings_at_stop") or {}
    return ", ".join(fmt(r.get(k), 2) for k in ("a", "b", "c", "d", "e"))


def from_when_cell(rec: dict) -> str:
    f = rec.get("from_when") or {}
    return ", ".join(fmt(f.get(k)) for k in ("a", "b", "c", "d", "e"))


def label(name: str, rec: dict) -> str:
    return f"{rec['_variant']} {rec['_grid']} ({GRID_LABEL[rec['_grid']]}) {rec['_scheme']}"


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def precursor(_args: argparse.Namespace) -> None:
    """Item 2: the channel's layer along x and the two candidate lengths."""
    pre = json.loads((OUT / "precursor_precursor.json").read_text())
    rows = [
        [
            fmt(r["x_h"]),
            fmt(r["u_centre_over_u_in"], 4),
            fmt(r["delta_99_h"]),
            fmt(r["re_theta"], 4),
        ]
        for r in pre["rows"]
    ]
    print_table(
        ["x/H from the inlet", "U_centre / U_in", "delta_99 / H", "Re_theta"], rows
    )
    rows = []
    for key, quantity in (("delta", "delta_99 = 1.5 H"), ("theta", "Re_theta = 5,000")):
        reached = pre[
            "x_delta_99_reaches_1_5h" if key == "delta" else "x_re_theta_reaches_5000"
        ]
        cand = next(c for c in pre["candidates"] if c["l_up_h"] == pre[f"l_up_h_{key}"])
        rows.append(
            [
                quantity,
                fmt(reached, 4),
                fmt(pre[f"l_up_h_{key}"]),
                fmt(pre[f"centre_ratio_at_minus_4_{key}"], 4),
                fmt(pre[f"u_in_for_{key}"], 4),
                fmt(cand["rms"]),
                fmt(cand["max_abs"]),
            ]
        )
    print_table(
        [
            "Rule",
            "Reached at x/H",
            "L_up / H (plus 4 H, rounded up)",
            "U_centre / U_in at x/H = -4",
            "U_in for U_ref (m/s)",
            "RMS of u / U_centre against the x/H = -4 data",
            "Largest",
        ],
        rows,
    )
    print(
        f"Precursor: stop {pre['stop']} after {pre['outer']:,} outer iterations, "
        f"{pre['seconds']:.0f} s; inlet {pre['u_in']} m/s."
    )
    print()
    keep("precursor", pre)


def matrix(_args: argparse.Namespace) -> None:
    """Item 4: every flow run, one line each."""
    rows, data = [], []
    for name, rec in records().items():
        res = rec["residual"]
        rows.append(
            [
                label(name, rec),
                rec["_class"],
                rec["stop"],
                fmt(rec["outer"]),
                f"{fmt(rec['residual_min'], 3)} ({fmt(rec['residual_min_at'])}), {fmt(rec['residual_end'], 3)}",
                readings_cell(rec),
                from_when_cell(rec),
                f"{fmt(rec['max_speed_end'])}, {tuple(rec['at_end'])}",
                f"{fmt(rec['inner_mean'], 3)}, {fmt(rec['inner_max'])}",
                fmt(rec["cap_hits"]),
                f"{fmt(rec['k_sweeps_mean'], 3)}, {fmt(rec['eps_sweeps_mean'], 3)}; {fmt(rec['scalar_cap_hits'])}",
                f"{fmt(rec['seconds'], 4)}, {fmt(rec['seconds_per_outer'], 3)}",
                (rec.get("face_hash") or "-")[:16],
            ]
        )
        data.append(
            {
                "name": name,
                "class": rec["_class"],
                "stop": rec["stop"],
                "outer": rec["outer"],
                "residual_min": rec["residual_min"],
                "residual_end": rec["residual_end"],
                "residual_last": res[-1] if res else None,
                "readings_at_stop": rec.get("readings_at_stop"),
                "from_when": rec.get("from_when"),
                "max_speed_end": rec["max_speed_end"],
                "at_end": rec["at_end"],
                "inner_mean": rec["inner_mean"],
                "inner_max": rec["inner_max"],
                "cap_hits": rec["cap_hits"],
                "k_iterations_mean": rec["k_sweeps_mean"],
                "eps_iterations_mean": rec["eps_sweeps_mean"],
                "scalar_cap_hits": rec["scalar_cap_hits"],
                "seconds": rec["seconds"],
                "seconds_per_outer": rec["seconds_per_outer"],
                "face_hash": rec.get("face_hash"),
                "commit": rec["commit"],
            }
        )
    print_table(
        [
            "Run",
            "Class",
            "Stop",
            "Outer",
            "Residual: least (at), end",
            "Readings at stop (a), (b), (c), (d), (e)",
            "Holds from (a), (b), (c), (d), (e)",
            "Largest speed at end (m/s), cell",
            "CG per correction: mean, largest",
            "Cap hits",
            "k, eps CG iterations (mean); capped",
            "Wall (s), per outer",
            "Face hash",
        ],
        rows,
    )
    keep("matrix", data)


def walls(_args: argparse.Namespace) -> None:
    """Item 4: y+ on the three walls, the inflow check and the outlet reading."""
    rows, data = [], []
    for name, (rec, ev) in compared_records().items():
        y = ev["y_star"]
        pm = json.loads((OUT / f"{name}_pmean.json").read_text())
        rows.append(
            [
                label(name, rec),
                rec["_class"],
                fmt(ev["u_ref"]["computed"], 4),
                fmt(ev["u_ref"]["over_measured"], 4),
                fmt(ev["inflow"]["delta_99_over_h"]),
                fmt(ev["inflow"]["re_theta"], 4),
                f"{fmt(y['floor_downstream'])} ({fmt(y['floor_at_station'])})",
                f"{fmt(y['step_top'])} ({fmt(y['step_top_at_station'])})",
                f"{fmt(y['ceiling_upstream'])} ({fmt(y['ceiling_at_station'])})",
                fmt(y["floor"], 4),
                fmt(pm["p_mean_rate_last_200"], 2),
                fmt(rec["core"]["nu_t_over_nu_median"]),
                fmt(rec["core"]["nu_t_over_nu_all_max"], 4),
            ]
        )
        data.append(
            {
                "name": name,
                "class": rec["_class"],
                "u_ref": ev["u_ref"],
                "inflow": ev["inflow"],
                "y_star": y,
                "p_mean_rate_last_200": pm["p_mean_rate_last_200"],
                "p_mean_end": pm["p_mean_end"],
                "nu_t_over_nu_median": rec["core"]["nu_t_over_nu_median"],
                "nu_t_over_nu_max": rec["core"]["nu_t_over_nu_all_max"],
            }
        )
    print_table(
        [
            "Run",
            "Class",
            "U_centre at x/H = -4 (m/s)",
            "Over U_ref",
            "delta_99 / H",
            "Re_theta",
            "y+ floor downstream of x_r: median (at x/H = 10)",
            "y+ step top: median (at x/H = -4)",
            "y+ ceiling upstream: median (at x/H = -4)",
            "Scalable floor",
            "Mean pressure drift, last 200 outer (Pa per iteration)",
            "nu_t / nu: median",
            "largest",
        ],
        rows,
    )
    keep("walls", data)


def reattachment(_args: argparse.Namespace) -> None:
    """Item 5: x_r on every converged run; the data; the published runs; the study."""
    rows, data = [], {"runs": [], "published": backstep.PUBLISHED_X_R}
    for name, (rec, ev) in compared_records().items():
        x_r = ev["x_r_over_h"]
        dep = None if x_r is None else x_r - backstep.X_R_MEASURED
        rows.append(
            [
                label(name, rec),
                rec["_class"],
                fmt(x_r, 4),
                fmt(dep, 3),
                "-" if dep is None else f"{100 * dep / backstep.X_R_MEASURED:+.1f}%",
            ]
        )
        data["runs"].append({"name": name, "class": rec["_class"], "x_r_over_h": x_r})
    rows.append(
        [
            "Driver and Seegmiller (1985), measured",
            "-",
            f"{backstep.X_R_MEASURED} +/- {backstep.X_R_UNCERTAINTY}",
            "0",
            "0",
        ]
    )
    for model, x_r in backstep.PUBLISHED_X_R.items():
        dep = x_r - backstep.X_R_MEASURED
        rows.append(
            [
                f"NPARC archive: {model}",
                "published",
                fmt(x_r, 3),
                fmt(dep, 3),
                f"{100 * dep / backstep.X_R_MEASURED:+.1f}%",
            ]
        )
    print_table(["Run", "Class", "x_r / H", "Minus measured (H)", "Relative"], rows)
    print(
        "The Chien k-epsilon rows are low-Reynolds-number k-epsilon runs integrated to "
        "the wall on a y+ = 1 grid, not the standard model with wall functions this "
        "solver runs."
    )
    print()

    # The resolution study: standard upwind on the three grids.
    study = {}
    for rec, ev in compared_records().values():
        if rec["_variant"] == "standard" and rec["_scheme"] == "upwind":
            study[rec["_grid"]] = (ev["x_r_over_h"], rec["_class"])
    rows = [
        [f"{g} ({GRID_LABEL[g]})", study[g][1], fmt(study[g][0], 4)]
        for g in GRIDS
        if g in study
    ]
    order = spread = None
    values = [study[g][0] for g in GRIDS if g in study and study[g][0] is not None]
    if len(values) == 3:
        d1, d2 = values[0] - values[1], values[1] - values[2]
        if d1 * d2 > 0:
            order = math.log(abs(d1) / abs(d2)) / math.log(REFINEMENT)
        spread = max(values) - min(values)
    fine_pair = abs(values[-1] - values[-2]) if len(values) >= 2 else None
    extrapolated = None
    if order is not None and order > 0:
        extrapolated = values[-1] + (values[-1] - values[-2]) / (
            REFINEMENT**order - 1.0
        )
        rows.append(
            ["extrapolated (observed order, ratio 1.5)", "-", fmt(extrapolated, 4)]
        )
    print_table(["Grid", "Class", "x_r / H (standard, upwind)"], rows)
    print(
        f"Observed order of x_r over the three grids (ratio {REFINEMENT} per axis): "
        f"{'not monotone' if order is None and len(values) == 3 else fmt(order)}; "
        f"spread {fmt(spread)} H; medium against fine {fmt(fine_pair)} H."
    )
    print()
    data["study"] = {
        "x_r": {g: study[g][0] for g in study},
        "classes": {g: study[g][1] for g in study},
        "observed_order": order,
        "spread_h": spread,
        "medium_fine_difference_h": fine_pair,
        "extrapolated": extrapolated,
    }
    keep("reattachment", data)


def profiles(_args: argparse.Namespace) -> None:
    """Item 5: each converged run at each station against the data."""
    rows, data = [], []
    stations = [
        str(s) for s in (backstep.REFERENCE_STATION, *backstep.PROFILE_STATIONS)
    ]
    for name, (rec, ev) in compared_records().items():
        row = [label(name, rec), rec["_class"]]
        entry = {"name": name, "class": rec["_class"], "profiles": {}}
        for s in stations:
            e = ev["profiles"][s]
            row.append(f"{fmt(e['max_abs'])} / {fmt(e['rms'])} ({e['stations']})")
            entry["profiles"][s] = e
        rows.append(row)
        data.append(entry)
    print_table(
        ["Run", "Class"]
        + [f"x/H = {s}: largest / RMS of u / U_ref (heights)" for s in stations],
        rows,
    )
    keep("profiles", data)


def coefficients(_args: argparse.Namespace) -> None:
    """Item 5: Cf and Cp along the bottom wall against the data."""
    rows, data = [], []
    for name, (rec, ev) in compared_records().items():
        cf, cp = ev["cf"], ev["cp"]
        rows.append(
            [
                label(name, rec),
                rec["_class"],
                fmt(cf["max_abs"], 3),
                fmt(cf["rms"], 3),
                fmt(cf["stations"]),
                fmt(cp["max_abs"], 3),
                fmt(cp["rms"], 3),
                fmt(cp["stations"]),
            ]
        )
        data.append({"name": name, "class": rec["_class"], "cf": cf, "cp": cp})
    print_table(
        [
            "Run",
            "Class",
            "Cf: largest difference",
            "RMS",
            "stations",
            "Cp (shifted to zero at x/H = 37.5): largest difference",
            "RMS",
            "stations",
        ],
        rows,
    )
    keep("coefficients", data)


def station_values(name: str, rec: dict) -> dict[float, np.ndarray]:
    """u / U_ref at the data's heights of every station, from a record's faces."""
    arrays = np.load(OUT / f"{name}.npz")
    config = SimConfig.from_dict(rec["raw"])
    mesh = Mesh(config)
    u_ref = backstep.centreline_speed(mesh, config, arrays["u_faces"])
    return backstep.profile_values(mesh, config, arrays["u_faces"], u_ref)


def schemes(_args: argparse.Namespace) -> None:
    """Item 5: upwind against umist on the medium grid, per station and in x_r."""
    recs = compared_records()
    pair = {
        rec["_scheme"]: (name, rec, ev)
        for name, (rec, ev) in recs.items()
        if rec["_grid"] == "656x60" and rec["_variant"] == "standard"
    }
    data: dict = {"available": sorted(pair)}
    if len(pair) < 2:
        print("The medium grid does not have both schemes compared.")
        print()
        keep("schemes", data)
        return
    up, um = pair["upwind"], pair["umist"]
    v_up, v_um = station_values(up[0], up[1]), station_values(um[0], um[1])
    rows = []
    data["stations"] = {}
    for station, profile in backstep.PROFILES.items():
        meas = np.array(profile.u_over_uref)
        between = backstep.errors(v_up[station], v_um[station])
        e_up = backstep.errors(v_up[station], meas)
        e_um = backstep.errors(v_um[station], meas)
        larger = max(e_up.max_abs, e_um.max_abs)
        ratio = between.max_abs / larger
        rows.append(
            [
                fmt(station),
                fmt(between.max_abs),
                fmt(between.rms),
                fmt(e_up.max_abs),
                fmt(e_um.max_abs),
                fmt(ratio),
                "holds" if ratio < D_RATIO else "fails",
            ]
        )
        data["stations"][str(station)] = {
            "between_max": between.max_abs,
            "between_rms": between.rms,
            "upwind_max": e_up.max_abs,
            "umist_max": e_um.max_abs,
            "ratio": ratio,
        }
    print_table(
        [
            "x/H",
            "Largest difference between the schemes, u / U_ref",
            "RMS",
            "upwind's largest from the data",
            "umist's",
            "Ratio to the larger",
            f"Under {D_RATIO}",
        ],
        rows,
    )
    x_up, x_um = up[2]["x_r_over_h"], um[2]["x_r_over_h"]
    dx = None if None in (x_up, x_um) else abs(x_up - x_um)
    print(
        f"x_r / H: upwind {fmt(x_up, 4)}, umist {fmt(x_um, 4)}, difference {fmt(dx)} H "
        f"(prediction (d) asks under {D_TOL_H} H)."
    )
    print()
    data["x_r"] = {"upwind": x_up, "umist": x_um, "difference_h": dx}
    keep("schemes", data)


def predictions(_args: argparse.Namespace) -> None:
    """Section 6: (a) to (f) scored from the records."""
    pre = json.loads((OUT / "precursor_precursor.json").read_text())
    recs = compared_records()
    allrecs = records()
    rows, data = [], {}

    # (a) L_up between 60 and 105 H.
    l_up = pre["l_up_h_delta"]
    a = 60 <= l_up <= 105
    rows.append(
        [
            "(a)",
            f"L_up = {l_up} H (delta_99 rule); Re_theta rule {pre['l_up_h_theta']} H",
            "holds" if a else "fails",
        ]
    )
    data["a"] = {"l_up_h": l_up, "holds": a}

    # (b) standard upwind medium x_r in [5.0, 5.9].
    med = next(
        ((n, r, e) for n, (r, e) in recs.items() if n == "standard_656x60_upwind"), None
    )
    if med is None or med[2]["x_r_over_h"] is None:
        rows.append(
            [
                "(b)",
                "no compared medium standard upwind run, or no reattachment",
                "not scorable",
            ]
        )
        data["b"] = {"holds": None}
    else:
        x_r = med[2]["x_r_over_h"]
        b = B_RANGE[0] <= x_r <= B_RANGE[1]
        verdict = (
            "holds" if b else ("fails long" if x_r > B_RANGE[1] else "fails short")
        )
        rows.append(
            [
                "(b)",
                f"x_r / H = {x_r:.3f} on the medium grid ({med[1]['_class']}); band {B_RANGE}",
                verdict,
            ]
        )
        data["b"] = {"x_r_over_h": x_r, "holds": b, "class": med[1]["_class"]}

    # (c) medium and fine differ by under 0.2 H.
    study = {
        r["_grid"]: e["x_r_over_h"]
        for _n, (r, e) in recs.items()
        if r["_variant"] == "standard" and r["_scheme"] == "upwind"
    }
    if (
        "656x60" in study
        and "984x90" in study
        and None not in (study["656x60"], study["984x90"])
    ):
        d = abs(study["656x60"] - study["984x90"])
        rows.append(
            [
                "(c)",
                f"medium {study['656x60']:.3f}, fine {study['984x90']:.3f}: {d:.3f} H",
                "holds" if d < C_TOL_H else "fails",
            ]
        )
        data["c"] = {"difference_h": d, "holds": d < C_TOL_H}
    else:
        rows.append(
            ["(c)", "the fine or the medium run is not compared", "not scorable"]
        )
        data["c"] = {"holds": None}

    # (d) the schemes within 0.2 H in x_r and under half the larger distance per station.
    sch = (
        json.loads((OUT / "tables_schemes.json").read_text())
        if (OUT / "tables_schemes.json").exists()
        else {}
    )
    if "stations" in sch and sch["x_r"]["difference_h"] is not None:
        ratios = {
            s: v["ratio"]
            for s, v in sch["stations"].items()
            if float(s) in backstep.PROFILE_STATIONS
        }
        d_ok = sch["x_r"]["difference_h"] < D_TOL_H and all(
            v < D_RATIO for v in ratios.values()
        )
        rows.append(
            [
                "(d)",
                f"x_r differs by {sch['x_r']['difference_h']:.3f} H; ratios {', '.join(f'{s}: {v:.2f}' for s, v in ratios.items())}",
                "holds" if d_ok else "fails",
            ]
        )
        data["d"] = {
            "difference_h": sch["x_r"]["difference_h"],
            "ratios": ratios,
            "holds": d_ok,
        }
    else:
        rows.append(
            ["(d)", "both schemes are not compared on the medium grid", "not scorable"]
        )
        data["d"] = {"holds": None}

    # (e) at x/H = 1 and 4, outside the recirculation (above the step's top), within 0.1 U_ref.
    if med is not None:
        values = station_values(med[0], med[1])
        worst = {}
        for station in E_STATIONS:
            profile = backstep.PROFILES[station]
            y = np.array(profile.y_over_h)
            diff = np.abs(values[station] - np.array(profile.u_over_uref))
            worst[station] = float(diff[y >= E_ABOVE_H].max())
        e = all(w <= E_TOL for w in worst.values())
        rows.append(
            [
                "(e)",
                f"largest |u / U_ref - data| at y/H >= {E_ABOVE_H}: "
                + ", ".join(f"x/H = {s}: {w:.3f}" for s, w in worst.items()),
                "holds" if e else "fails",
            ]
        )
        data["e"] = {"worst_above_step": worst, "holds": e}
    else:
        rows.append(["(e)", "no compared medium standard upwind run", "not scorable"])
        data["e"] = {"holds": None}

    # (f) RNG: if it converges, longer than standard.
    rng = allrecs.get("rng_656x60_upwind")
    if rng is None:
        rows.append(["(f)", "RNG not run", "not scorable"])
        data["f"] = {"holds": None}
    elif rng["stop"] != CONVERGED:
        rows.append(
            [
                "(f)",
                f"RNG {rng['_class']} at {rng['outer']:,} outer iterations; no prediction on convergence",
                "not scorable (did not converge)",
            ]
        )
        data["f"] = {"class": rng["_class"], "holds": None}
    else:
        x_rng = evaluation("rng_656x60_upwind", rng)["x_r_over_h"]
        x_std = None if med is None else med[2]["x_r_over_h"]
        f = None if None in (x_rng, x_std) else x_rng > x_std
        rows.append(
            [
                "(f)",
                f"RNG converged: x_r / H {fmt(x_rng, 4)} against standard {fmt(x_std, 4)}",
                "holds" if f else ("fails" if f is not None else "not scorable"),
            ]
        )
        data["f"] = {"x_r_rng": x_rng, "x_r_standard": x_std, "holds": f}

    print_table(["Prediction", "Measured", "Verdict"], rows)
    keep("predictions", data)


def timing(_args: argparse.Namespace) -> None:
    """The run times, beside other runs."""
    rows, data = [], []
    for name, rec in records().items():
        rows.append(
            [
                label(name, rec),
                fmt(rec["outer"]),
                fmt(rec["seconds"], 4),
                fmt(rec["seconds"] / 3600.0, 3),
                fmt(rec["seconds_per_outer"], 3),
                fmt(rec["inner_mean"], 3),
                ", ".join(
                    f"{k} {v:.0f}" for k, v in (rec.get("stage_seconds") or {}).items()
                ),
            ]
        )
        data.append(
            {
                "name": name,
                "outer": rec["outer"],
                "seconds": rec["seconds"],
                "seconds_per_outer": rec["seconds_per_outer"],
                "stage_seconds": rec.get("stage_seconds"),
            }
        )
    print_table(
        [
            "Run",
            "Outer",
            "Wall (s)",
            "Wall (h)",
            "Per outer (s)",
            "CG per correction (mean)",
            "Stage seconds",
        ],
        rows,
    )
    keep("timing", data)


MODES = (
    ("precursor", precursor),
    ("matrix", matrix),
    ("walls", walls),
    ("reattachment", reattachment),
    ("profiles", profiles),
    ("coefficients", coefficients),
    ("schemes", schemes),
    ("predictions", predictions),
    ("timing", timing),
)


def everything(args: argparse.Namespace) -> None:
    for name, mode in MODES:
        print(f"<!-- tables52 {name} -->")
        mode(args)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    for name, fn in MODES:
        sub.add_parser(name).set_defaults(func=fn)
    sub.add_parser("all").set_defaults(func=everything)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
