"""The report's tables from the prompt 55 records.

Usage:
    python tables55.py matrix       item 3: every run, classified by step 5's rule
    python tables55.py centreline   item 4: the centreline against the data per compared run
    python tables55.py halfwidth    item 4: the half-width and the spreading rate
    python tables55.py lateral      item 4: the lateral profiles per height
    python tables55.py turbulence   item 4: k on the axis, the stagnation point, y+
    python tables55.py schemes      item 4: upwind against umist on the fine grid
    python tables55.py grids        item 4: the coarse grid against the fine
    python tables55.py width        item 2: the 68 w domain against the 100 w domain
    python tables55.py predictions  section 6: (a) to (j) scored from the records
    python tables55.py timing       the run times
    python tables55.py all          everything above

Reads results/builder55/NAME.json and NAME.npz (impinging55.py), prints
markdown tables, and keeps every number it printed in
results/builder55/tables_MODE.json. The classification is tables46's
(imported): diverged, converged, growing, bounded with step 0's
sub-classes, or a positivity error, its growing threshold scaled to the
case's speed. The comparison numbers are validation.impinging_jet.
evaluate's, recomputed here from each record's final faces and k, so the
tables and the harness row read the same code.
"""

import argparse
import json
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
from validation import impinging_jet as ij  # noqa: E402

OUT = ROOT / "results" / "builder55"
tables46.OUT = OUT
# tables46's growing threshold is 5 m/s, ten times a room's supply speed;
# the jet issues at V0, so the same ratio is ten times V0 (the runner's
# divergence guard).
tables46.GROWING_SPEED = 10.0 * ij.V0
GRIDS = ("272x90", "408x135")
GRID_LABEL = {"272x90": "coarse", "408x135": "fine"}
SCHEMES = ("upwind", "umist")
VARIANTS = ("standard", "rng")
NAME = re.compile(
    r"^(?P<variant>standard|rng)_(?P<grid>\d+x\d+)_(?P<scheme>upwind|umist)(?P<tag>_w100)?$"
)
CONVERGED = "error_estimate_and_continuity"
AT_CAP_COMPARED = "bounded"
BASE = "standard_272x90_upwind"
WIDE = "standard_272x90_upwind_w100"
FINE = "standard_408x135_upwind"
FINE_UMIST = "standard_408x135_umist"
RNG = "rng_408x135_upwind"
# Prediction thresholds (section 3 of the report).
C_TOL = 0.1
D_RATIO = 1.5
E_SCHEME_SHARE = 0.5
F_CENTRE_TOL = 0.02
F_WIDTH_TOL = 0.05
G_CORE_RANGE = (6.0, 9.0)
H_RANGE = (0.10, 0.12)
I_RANGE = (5.0, 30.0)
LATERAL_HEIGHTS = ("4.8", "5.9", "7.9", "12.1", "14.9", "20.1")


def keep(mode: str, data: object) -> None:
    """The printed numbers, as JSON beside the records."""
    (OUT / f"tables_{mode}.json").write_text(json.dumps(data, indent=1, default=float))


def records() -> dict[str, dict]:
    """Every flow record of item 3 and the width check, keyed by name."""
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
        rec["_tag"] = m["tag"] or ""
        cls, sub = classify(rec)
        rec["_class"] = cls + (f" ({sub})" if sub else "")
        found[path.stem] = rec
    return dict(
        sorted(
            found.items(),
            key=lambda kv: (
                kv[1]["_tag"] != "",
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
    """validation.impinging_jet.evaluate on a record's final fields, cached beside it."""
    cache = OUT / f"eval_{name}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    arrays = np.load(OUT / f"{name}.npz")
    config = SimConfig.from_dict(rec["raw"])
    mesh = Mesh(config)
    report = ij.evaluate(mesh, config, arrays["v_faces"], arrays["k"])
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
    width = f", {rec.get('width_w', 68):.0f} w wide" if rec["_tag"] else ""
    return f"{rec['_variant']} {rec['_grid']} ({GRID_LABEL[rec['_grid']]}) {rec['_scheme']}{width}"


def status(rec: dict) -> str:
    """How a compared run's numbers are labelled."""
    return "converged" if rec["stop"] == CONVERGED else f"at the cap, {rec['_class']}"


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def matrix(_args: argparse.Namespace) -> None:
    """Item 3: every run, one line each."""
    rows, data = [], []
    for name, rec in records().items():
        pmean = OUT / f"{name}_pmean.json"
        drift = (
            json.loads(pmean.read_text())["p_mean_rate_last_200"]
            if pmean.exists()
            else None
        )
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
                fmt(drift, 2),
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
                "residual_min_at": rec["residual_min_at"],
                "residual_end": rec["residual_end"],
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
                "p_mean_rate_last_200": drift,
                "positivity": rec.get("positivity"),
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
            "p_mean drift (Pa per outer, last 200)",
            "Wall (s), per outer",
            "Face hash",
        ],
        rows,
    )
    keep("matrix", data)


def centreline(_args: argparse.Namespace) -> None:
    """Item 4: the centreline against the data per compared run."""
    rows, data = [], []
    rows.append(
        [
            "data (digitised)",
            "-",
            fmt(ij.potential_core_w(ij.CENTRELINE_Y, ij.CENTRELINE_V), 3),
            fmt(ij.first_crossing(ij.CENTRELINE_Y, ij.CENTRELINE_V, 0.5), 3),
            fmt(ij.decay_rate(ij.CENTRELINE_Y, ij.CENTRELINE_V), 3),
            "-",
            "-",
            "-",
            "-",
        ]
    )
    rows.append(
        [
            "paper's text",
            "-",
            fmt(ij.STATED["potential_core_w"]),
            fmt(ij.STATED["half_velocity_h"]),
            f"{ij.STATED['decay_rate'][0]} to {ij.STATED['decay_rate'][1]}",
            "-",
            "-",
            "-",
            "-",
        ]
    )
    for name, (rec, ev) in compared_records().items():
        c = ev["centreline"]

        def cell(part: str, c: dict = c) -> str:
            e = c[part]
            return f"{fmt(e['max_abs'], 3)}, {fmt(e['rms'], 3)}, {fmt(e['mean'], 2)} ({e['stations']})"

        rows.append(
            [
                label(name, rec),
                status(rec),
                fmt(ev["potential_core_w"]["computed"], 3),
                fmt(ev["half_velocity_h"]["computed"], 3),
                fmt(ev["decay_rate"]["computed"], 3),
                cell("all"),
                cell("free_jet"),
                cell("free_jet_narrow"),
                cell("impingement"),
            ]
        )
        data.append(
            {
                "name": name,
                "status": status(rec),
                "potential_core_w": ev["potential_core_w"],
                "half_velocity_h": ev["half_velocity_h"],
                "decay_rate": ev["decay_rate"],
                "all": c["all"],
                "free_jet": c["free_jet"],
                "free_jet_narrow": c["free_jet_narrow"],
                "impingement": c["impingement"],
            }
        )
    print_table(
        [
            "Run",
            "Fields",
            "Potential core (w)",
            "Half-velocity (y/h)",
            "K_u",
            "Whole line: largest, RMS, mean (stations)",
            "Free jet y/h <= 0.70",
            "Nearest the nozzle y/h <= 0.30",
            "Impingement y/h > 0.88",
        ],
        rows,
    )
    keep("centreline", data)


def halfwidth(_args: argparse.Namespace) -> None:
    """Item 4: delta_0.5 / w at the dot heights and the spreading rate."""
    heights = ij.HALF_WIDTH_Y
    headers = (
        ["Run", "Fields"] + [f"y/w {h:.1f}" for h in heights] + ["Largest, RMS", "K_y"]
    )
    rows = [
        ["data (dots)", "-"]
        + [fmt(d, 3) for d in ij.HALF_WIDTH_DELTA]
        + ["-", fmt(ij.spreading_rate(ij.HALF_WIDTH_Y, ij.HALF_WIDTH_DELTA), 3)],
        ["paper's text", "-"]
        + ["-"] * len(heights)
        + ["-", fmt(ij.STATED["spreading_rate"])],
    ]
    data = []
    for name, (rec, ev) in compared_records().items():
        hw = ev["half_width"]
        e = hw["errors"]
        rows.append(
            [label(name, rec), status(rec)]
            + [fmt(d, 3) for d in hw["computed"]]
            + [
                "-" if e is None else f"{fmt(e['max_abs'], 3)}, {fmt(e['rms'], 3)}",
                fmt(hw["spreading_rate"]["computed"], 3),
            ]
        )
        data.append({"name": name, "status": status(rec), **hw})
    print_table(headers, rows)
    keep("halfwidth", data)


def lateral(_args: argparse.Namespace) -> None:
    """Item 4: the self-similar lateral profiles per height, and the lateral I_uv at 4.8."""
    headers = (
        ["Run", "Fields"]
        + [
            f"y'/w {h} ({'digitised' if h == '4.8' else 'Gaussian'})"
            for h in LATERAL_HEIGHTS
        ]
        + ["I_uv at 4.8"]
    )
    rows, data = [], []
    for name, (rec, ev) in compared_records().items():
        cells = []
        for h in LATERAL_HEIGHTS:
            entry = ev["lateral"][h]
            e = entry["errors"]
            cells.append(
                "-"
                if e is None
                else f"{fmt(e['max_abs'], 3)}, {fmt(e['rms'], 3)}; delta {fmt(entry['delta_over_w'], 3)} w"
            )
        li = ev["lateral_intensity_48"]["errors"]
        cells.append(
            "-" if li is None else f"{fmt(li['max_abs'], 3)}, {fmt(li['rms'], 3)}"
        )
        rows.append([label(name, rec), status(rec), *cells])
        data.append(
            {
                "name": name,
                "status": status(rec),
                "lateral": {
                    h: {
                        k: v
                        for k, v in ev["lateral"][h].items()
                        if k not in ("computed", "reference", "x_over_delta")
                    }
                    for h in LATERAL_HEIGHTS
                },
                "lateral_intensity_48": ev["lateral_intensity_48"]["errors"],
            }
        )
    print_table(headers, rows)
    keep("lateral", data)


def turbulence(_args: argparse.Namespace) -> None:
    """Item 4: k on the axis against the PIV estimate, the stagnation point, the plate's y+."""
    rows, data = [], []
    for name, (rec, ev) in compared_records().items():
        ka, st, ys = ev["k_axis"], ev["stagnation"], ev["y_star"]
        rows.append(
            [
                label(name, rec),
                status(rec),
                f"{fmt(ka['errors_isotropic']['max_abs'], 3)}, {fmt(ka['errors_isotropic']['rms'], 3)}, {fmt(ka['errors_isotropic']['mean'], 2)}",
                fmt(ka["ratio_isotropic_median"], 3),
                f"{fmt(st['computed_at_data_station'], 3)} vs {fmt(st['data_isotropic'], 3)} (in-plane {fmt(st['data_in_plane'], 3)}) at y/h {fmt(st['data_station_y_over_h'], 3)}",
                f"{fmt(st['ratio_isotropic'], 3)}, {fmt(st['ratio_in_plane'], 3)}",
                fmt(st["first_cell"], 3),
                f"{fmt(st['impingement_max'], 3)} at y/h {fmt(st['impingement_max_y_over_h'], 3)}",
                f"{fmt(ys['median'], 3)}, {fmt(ys['at_axis'], 3)}, {fmt(ys['largest'], 3)}; floor {fmt(ys['floor'], 4)}",
            ]
        )
        data.append(
            {
                "name": name,
                "status": status(rec),
                "k_axis": {k: v for k, v in ka.items() if not isinstance(v, list)},
                "stagnation": st,
                "y_star": ys,
            }
        )
    print_table(
        [
            "Run",
            "Fields",
            "k/V0^2 on the axis vs 0.75 I_uv^2: largest, RMS, mean",
            "Median ratio model / estimate",
            "Stagnation: model vs estimate (k/V0^2)",
            "Ratio to isotropic, to in-plane",
            "First cell above the plate",
            "Largest on the axis in the impingement region",
            "Plate y+: median within 10 w, at the axis, largest; floor",
        ],
        rows,
    )
    keep("turbulence", data)


def quantities(ev: dict) -> dict[str, float | None]:
    """The scalar quantities two runs are compared on."""
    out: dict[str, float | None] = {
        "centreline_rms_all": ev["centreline"]["all"]["rms"],
        "centreline_rms_free_jet": ev["centreline"]["free_jet"]["rms"],
        "centreline_rms_impingement": ev["centreline"]["impingement"]["rms"],
        "potential_core_w": ev["potential_core_w"]["computed"],
        "decay_rate": ev["decay_rate"]["computed"],
        "half_width_rms": None
        if ev["half_width"]["errors"] is None
        else ev["half_width"]["errors"]["rms"],
        "spreading_rate": ev["half_width"]["spreading_rate"]["computed"],
        "k_axis_rms": ev["k_axis"]["errors_isotropic"]["rms"],
        "stagnation_k": ev["stagnation"]["computed_at_data_station"],
    }
    for h in LATERAL_HEIGHTS:
        e = ev["lateral"][h]["errors"]
        out[f"lateral_{h}_rms"] = None if e is None else e["rms"]
    return out


def data_values() -> dict[str, float | None]:
    """The data's own value of each quantity, where it has one (the rest are distances already)."""
    return {
        "potential_core_w": ij.potential_core_w(ij.CENTRELINE_Y, ij.CENTRELINE_V),
        "decay_rate": ij.decay_rate(ij.CENTRELINE_Y, ij.CENTRELINE_V),
        "spreading_rate": ij.spreading_rate(ij.HALF_WIDTH_Y, ij.HALF_WIDTH_DELTA),
        "stagnation_k": ij.K_ISOTROPIC
        * float(ij.INTENSITY_I[int(np.argmax(ij.INTENSITY_Y))]) ** 2,
    }


def pair_table(mode: str, first: str, second: str, title: str) -> None:
    """Two runs on every quantity: each's value, their difference, each's distance from the data."""
    recs = compared_records()
    if first not in recs or second not in recs:
        print(f"{mode}: {first} or {second} is not a compared record")
        keep(mode, {"missing": [n for n in (first, second) if n not in recs]})
        return
    a, b = quantities(recs[first][1]), quantities(recs[second][1])
    ref = data_values()
    rows, data = [], []
    for key in a:
        va, vb = a[key], b[key]
        if va is None or vb is None:
            rows.append([key, fmt(va, 4), fmt(vb, 4), "-", "-", "-", "-"])
            data.append({"quantity": key, "first": va, "second": vb})
            continue
        diff = abs(va - vb)
        if key in ref:
            da, db = abs(va - ref[key]), abs(vb - ref[key])
        else:
            da, db = va, vb  # an RMS is a distance already
        larger = max(da, db)
        share = diff / larger if larger > 0 else None
        rows.append(
            [
                key,
                fmt(va, 4),
                fmt(vb, 4),
                fmt(diff, 3),
                fmt(da, 3),
                fmt(db, 3),
                fmt(share, 3),
            ]
        )
        data.append(
            {
                "quantity": key,
                "first": va,
                "second": vb,
                "difference": diff,
                "first_from_data": da,
                "second_from_data": db,
                "share": share,
            }
        )
    print_table(
        [
            "Quantity",
            title.split(" vs ")[0],
            title.split(" vs ")[1],
            "|difference|",
            "first from data",
            "second from data",
            "difference / larger distance",
        ],
        rows,
    )
    keep(mode, {"first": first, "second": second, "rows": data})


def schemes(_args: argparse.Namespace) -> None:
    """Item 4: upwind against umist on the fine grid."""
    pair_table("schemes", FINE, FINE_UMIST, "fine upwind vs fine umist")


def grids(_args: argparse.Namespace) -> None:
    """Item 4: the coarse grid against the fine, both upwind."""
    pair_table("grids", BASE, FINE, "coarse upwind vs fine upwind")


def width(_args: argparse.Namespace) -> None:
    """Item 2: the 68 w domain against the 100 w domain on the coarse grid."""
    recs = compared_records()
    if BASE not in recs or WIDE not in recs:
        print(f"width: {BASE} or {WIDE} is not a compared record")
        keep("width", {"missing": [n for n in (BASE, WIDE) if n not in recs]})
        return
    a, b = recs[BASE][1], recs[WIDE][1]
    centre = np.abs(
        np.array(a["centreline"]["computed"]) - np.array(b["centreline"]["computed"])
    )
    held = [
        (x, y)
        for x, y in zip(
            a["half_width"]["computed"], b["half_width"]["computed"], strict=True
        )
        if x is not None and y is not None
    ]
    hw = (
        np.abs(np.array([x for x, _ in held]) - np.array([y for _, y in held]))
        if held
        else np.array([])
    )
    la, lb = a["lateral"]["4.8"]["computed"], b["lateral"]["4.8"]["computed"]
    lat = np.abs(np.array(la) - np.array(lb)) if la and lb else np.array([])
    k_axis = np.abs(
        np.array(a["k_axis"]["computed"]) - np.array(b["k_axis"]["computed"])
    )
    data = {
        "centreline_max_abs": float(centre.max()),
        "centreline_rms": float(np.sqrt(np.mean(centre**2))),
        "half_width_max_abs_w": float(hw.max()) if hw.size else None,
        "lateral_48_max_abs": float(lat.max()) if lat.size else None,
        "k_axis_max_abs": float(k_axis.max()),
        "stagnation_k_68": a["stagnation"]["computed_at_data_station"],
        "stagnation_k_100": b["stagnation"]["computed_at_data_station"],
        "potential_core_68": a["potential_core_w"]["computed"],
        "potential_core_100": b["potential_core_w"]["computed"],
        "status_68": status(recs[BASE][0]),
        "status_100": status(recs[WIDE][0]),
    }
    print_table(
        ["Quantity", "68 w minus 100 w, largest over the stations"],
        [
            [
                "centreline V_0,y / V0 (51 stations)",
                f"{fmt(data['centreline_max_abs'], 3)} (RMS {fmt(data['centreline_rms'], 3)})",
            ],
            [
                "half-width delta_0.5 / w (11 heights)",
                fmt(data["half_width_max_abs_w"], 3),
            ],
            [
                "lateral V / V_0,y at 4.8 (19 stations)",
                fmt(data["lateral_48_max_abs"], 3),
            ],
            ["k / V0^2 on the axis (40 stations)", fmt(data["k_axis_max_abs"], 4)],
            [
                "k / V0^2 at the stagnation station",
                f"{fmt(data['stagnation_k_68'], 4)} vs {fmt(data['stagnation_k_100'], 4)}",
            ],
            [
                "potential core (w)",
                f"{fmt(data['potential_core_68'], 3)} vs {fmt(data['potential_core_100'], 3)}",
            ],
            ["fields", f"{data['status_68']} vs {data['status_100']}"],
        ],
    )
    keep("width", data)


def predictions(_args: argparse.Namespace) -> None:
    """Section 6: (a) to (j) scored from the records."""
    recs = records()
    comp = compared_records()
    rows, data = [], []

    def add(key: str, claim: str, verdict: str, evidence: str, **extra: object) -> None:
        rows.append([key, claim, verdict, evidence])
        data.append(
            {"prediction": key, "verdict": verdict, "evidence": evidence, **extra}
        )

    def conv(name: str) -> bool:
        return name in recs and recs[name]["stop"] == CONVERGED

    # (a)
    both = conv(BASE) and conv(FINE)
    add(
        "(a)",
        "standard converges on both grids",
        "HOLDS" if both else "FAILS",
        "; ".join(
            f"{n}: {recs[n]['stop']} ({recs[n]['_class']}) at {recs[n]['outer']:,}"
            for n in (BASE, FINE)
            if n in recs
        ),
    )
    # (b)
    if conv(RNG):
        k_rng = comp[RNG][1]["stagnation"]["computed_at_data_station"]
        k_std = (
            comp[FINE][1]["stagnation"]["computed_at_data_station"]
            if FINE in comp
            else None
        )
        ok = k_std is not None and k_rng < k_std
        add(
            "(b)",
            "if RNG converges, its stagnation k is below standard's",
            "HOLDS" if ok else "FAILS",
            f"RNG {k_rng:.4g} vs standard {k_std}",
        )
    else:
        add(
            "(b)",
            "if RNG converges, its stagnation k is below standard's",
            "NOT SCORABLE",
            f"RNG: {recs[RNG]['stop']} ({recs[RNG]['_class']}) at {recs[RNG]['outer']:,}"
            if RNG in recs
            else "no RNG record",
        )
    # (c)
    for name in (BASE, FINE):
        if name in comp:
            c = comp[name][1]["centreline"]
            ok = c["free_jet"]["max_abs"] < C_TOL
            add(
                "(c)",
                f"standard within 0.1 V0 of the data over the free-jet part ({name})",
                "HOLDS" if ok else "FAILS",
                f"largest |difference| {c['free_jet']['max_abs']:.3f} over y/h <= 0.70 ({c['free_jet']['stations']} stations); {c['free_jet_narrow']['max_abs']:.3f} over y/h <= 0.30 ({c['free_jet_narrow']['stations']})",
            )
    # (d)
    for name in (BASE, FINE):
        if name in comp:
            st = comp[name][1]["stagnation"]
            ok = st["ratio_isotropic"] > D_RATIO
            add(
                "(d)",
                f"standard overpredicts the stagnation k by more than 50% ({name})",
                "HOLDS" if ok else "FAILS",
                f"model {st['computed_at_data_station']:.4g} vs estimate {st['data_isotropic']:.4g} at y/h {st['data_station_y_over_h']}: ratio {st['ratio_isotropic']:.3f} (in-plane {st['ratio_in_plane']:.3f}); first cell {st['first_cell']:.4g}, largest in the impingement region {st['impingement_max']:.4g}",
            )
    # (e)
    for mode, first, second, rule in (
        ("schemes", FINE, FINE_UMIST, E_SCHEME_SHARE),
        ("grids", BASE, FINE, 1.0),
    ):
        if first in comp and second in comp:
            a, b = quantities(comp[first][1]), quantities(comp[second][1])
            ref = data_values()
            worst, failing = 0.0, []
            for key in a:
                if a[key] is None or b[key] is None:
                    continue
                diff = abs(a[key] - b[key])
                da, db = (
                    (abs(a[key] - ref[key]), abs(b[key] - ref[key]))
                    if key in ref
                    else (a[key], b[key])
                )
                larger = max(da, db)
                share = diff / larger if larger > 0 else 0.0
                worst = max(worst, share)
                if share >= rule:
                    failing.append(f"{key} {share:.2f}")
            add(
                "(e)",
                f"the {mode} differ by under {rule:g} of the larger distance from the data on every quantity",
                "HOLDS" if not failing else "FAILS",
                f"largest share {worst:.3f}"
                + (f"; failing: {', '.join(failing)}" if failing else ""),
            )
        else:
            add(
                "(e)",
                f"the {mode} clause",
                "NOT SCORABLE",
                f"{first} or {second} not compared",
            )
    # (f)
    w = (
        json.loads((OUT / "tables_width.json").read_text())
        if (OUT / "tables_width.json").exists()
        else {}
    )
    if "centreline_max_abs" in w:
        ok = w["centreline_max_abs"] < F_CENTRE_TOL and (
            w["half_width_max_abs_w"] is not None
            and w["half_width_max_abs_w"] < F_WIDTH_TOL
        )
        add(
            "(f)",
            "the 100 w domain moves the centreline by under 0.02 V0 and the half-width by under 0.05 w",
            "HOLDS" if ok else "FAILS",
            f"centreline {w['centreline_max_abs']:.4f}, half-width {w['half_width_max_abs_w']}",
        )
    else:
        add("(f)", "the width check", "NOT SCORABLE", "no width table")
    # (g)
    for name in (BASE, FINE):
        if name in comp:
            core = comp[name][1]["potential_core_w"]["computed"]
            mean_free = comp[name][1]["centreline"]["free_jet"]["mean"]
            ok = (
                core is not None
                and G_CORE_RANGE[0] <= core <= G_CORE_RANGE[1]
                and mean_free > 0
            )
            add(
                "(g)",
                f"potential core 6 to 9 w and the centreline above the data over the free-jet part ({name})",
                "HOLDS" if ok else "FAILS",
                f"core {core}, mean difference over the free-jet part {mean_free:+.4f}",
            )
    # (h)
    for name in (BASE, FINE):
        if name in comp:
            ky = comp[name][1]["half_width"]["spreading_rate"]["computed"]
            ok = ky is not None and H_RANGE[0] <= ky <= H_RANGE[1]
            add(
                "(h)",
                f"K_y between 0.10 and 0.12 ({name})",
                "HOLDS" if ok else "FAILS",
                f"K_y {ky}",
            )
    # (i)
    if BASE in comp:
        ys = comp[BASE][1]["y_star"]
        ok = I_RANGE[0] <= ys["median"] <= I_RANGE[1]
        fine_ys = comp[FINE][1]["y_star"] if FINE in comp else None
        add(
            "(i)",
            "the plate's first-node y+ between 5 and 30 on the coarse grid",
            "HOLDS" if ok else "FAILS",
            f"coarse median {ys['median']:.2f}, at the axis {ys['at_axis']:.2f}, largest {ys['largest']:.2f}"
            + (
                f"; fine median {fine_ys['median']:.2f}, largest {fine_ys['largest']:.2f}"
                if fine_ys
                else ""
            )
            + f"; floor {ys['floor']:.2f}",
        )
    # (j)
    if RNG in recs:
        add(
            "(j)",
            "RNG does not converge",
            "HOLDS" if not conv(RNG) else "FAILS",
            f"{recs[RNG]['stop']} ({recs[RNG]['_class']}) at {recs[RNG]['outer']:,}",
        )
    print_table(["Prediction", "Claim", "Verdict", "Evidence"], rows)
    keep("predictions", data)


def timing(_args: argparse.Namespace) -> None:
    """The run times."""
    rows, data = [], []
    for name, rec in records().items():
        rows.append(
            [
                label(name, rec),
                fmt(rec["nx"] * rec["ny"]),
                fmt(rec["outer"]),
                fmt(rec["seconds"], 4),
                fmt(rec["seconds_per_outer"], 3),
                fmt(rec["seconds"] / 3600.0, 3),
                rec["started"],
                ", ".join(
                    f"{k} {v:.0f}" for k, v in (rec.get("stage_seconds") or {}).items()
                ),
            ]
        )
        data.append(
            {
                "name": name,
                "cells": rec["nx"] * rec["ny"],
                "outer": rec["outer"],
                "seconds": rec["seconds"],
                "seconds_per_outer": rec["seconds_per_outer"],
                "started": rec["started"],
                "stage_seconds": rec.get("stage_seconds"),
            }
        )
    print_table(
        [
            "Run",
            "Cells",
            "Outer",
            "Wall (s)",
            "Per outer (s)",
            "Hours",
            "Started",
            "Stages (s)",
        ],
        rows,
    )
    keep("timing", data)


MODES = (
    ("matrix", matrix),
    ("centreline", centreline),
    ("halfwidth", halfwidth),
    ("lateral", lateral),
    ("turbulence", turbulence),
    ("schemes", schemes),
    ("grids", grids),
    ("width", width),
    ("predictions", predictions),
    ("timing", timing),
)


def everything(args: argparse.Namespace) -> None:
    for name, func in MODES:
        print(f"### {name}")
        func(args)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=[m for m, _ in MODES] + ["all"])
    args = parser.parse_args()
    if args.mode == "all":
        everything(args)
    else:
        dict(MODES)[args.mode](args)


if __name__ == "__main__":
    main()
