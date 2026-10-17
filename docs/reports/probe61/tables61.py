"""The tables of docs/reports/adr014_step0b.md, from the records of prompt 61.

Usage:
    python tables61.py           # rewrite the marked tables in the report
    python tables61.py --check   # exit 1 if a marked table differs

Reads results/builder61/ of the main checkout (hash_*.json from hash61.py, the
flow records vents_320x120, default_320x120 and closed_box from flow61.py, the
marches transport_*.json from transport61.py and plants.json from plants61.py)
and the records the hashes of record come from, as tables60.py names them
(results/builder54/, builder56/L0_320x120.json, builder59/rest.json). Each
table sits in the report between ``<!-- BEGIN tables61: NAME -->`` and
``<!-- END tables61: NAME -->``.

The metrics are prompt 58's (docs/reports/probe58/metrics58.md) on one case,
Sc_t 0.7 on 320x120, per class: M1 the deposition rate on the litho tool's and
the etch chamber's tops summed over the three sources; M2 the steady
concentration at near_door and hood_entry over each source's emission, summed
over sensors and sources; M3 the share of the emission removed through the
openings, averaged over the sources. M3 reads ``net_removal_over_source``:
outflow less inflow, which is prompt 58's figure in the default room (its
supply is clean) and, in the vented room, leaves out the room air the rack's
intake draws and its discharge returns.
"""

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "probe60"))

import hash61  # noqa: E402
import tables60  # noqa: E402

REPORT = HERE.parent / "adr014_step0b.md"
ROOMS = (("default_320x120", "default"), ("vents_320x120", "vents"))
SOURCES = ("S1", "S2", "S3")
CLASSES = ("2", "4")
SC = "0.7"
PROCESS_SURFACES = ("litho_tool top", "etch_chamber top")
OPERATOR_SENSORS = ("near_door", "hood_entry")
PREDICTION_D_SENSORS = ("near_door", "above_gap_1")
RATIO_LIMIT = 1.3


def load(path: Path) -> dict:
    """A JSON record; a missing one stops the run, naming it."""
    if not path.exists():
        sys.exit(f"record missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def table(headers: list[str], rows: list[list[str]]) -> str:
    """tables60's markdown table."""
    return tables60.table(headers, rows)


def hashes(results: Path) -> str:
    """Prediction (a): each case from rest at step 0b against its hash of record."""
    rows = []
    for label, source, origin in tables60.CASES:
        kept = load(results / source)
        run = load(results / "builder61" / f"hash_{label}.json")
        stop, outer = tables60.record_stop(kept)
        same = "yes" if run["face_hash"] == kept["face_hash"] else "NO"
        rows.append(
            [
                label,
                f"{kept['face_hash'][:16]} ({kept.get('commit', '?')}, {origin})",
                f"{stop} at {outer:,}",
                f"{run['face_hash'][:16]} ({run['commit']})",
                f"{run['stop']} at {run['outer']:,}",
                same,
            ]
        )
    return table(
        [
            "Case",
            "Hash of record (commit, source)",
            "Its stop",
            "At step 0b (commit)",
            "Stop",
            "Equal",
        ],
        rows,
    )


def flows(results: Path) -> str:
    """Prediction (b): the two rooms' solves, the vented one against the default."""
    default = load(results / "builder61" / "default_320x120.json")
    rows = []
    for name, room in ROOMS:
        r = load(results / "builder61" / f"{name}.json")
        ratio = r["outer"] / default["outer"]
        worst = max(abs(x) for x in r["signed_sums"])
        rows.append(
            [
                room,
                f"{r['stop']} at {r['outer']:,}",
                f"{ratio:.3f}",
                f"{r['wall_seconds']:,.0f}",
                f"{r['seconds_per_outer']:.3f}",
                f"{r['velocity_scale']:g}",
                f"{r['flux_scale']:.4g}",
                f"{worst / r['flux_scale']:.1e}",
                r["face_hash"][:16],
                str(r["concurrent_processes"]),
            ]
        )
    return table(
        [
            "Room (320x120)",
            "Stop",
            "Outer over the default's",
            "Wall (s)",
            "Wall per outer (s)",
            "Velocity scale (m/s)",
            "Flux scale F (kg/s per m)",
            "Worst signed sum over F",
            "Face hash",
            "Processes",
        ],
        rows,
    )


def verdict_b(results: Path) -> str:
    """Prediction (b)'s reading, one line."""
    default = load(results / "builder61" / "default_320x120.json")
    vents = load(results / "builder61" / "vents_320x120.json")
    ratio = vents["outer"] / default["outer"]
    held = vents["converged"] and ratio <= RATIO_LIMIT
    return (
        f"(b) {'holds' if held else 'fails'}: the vented room "
        f"{'stops by the rule' if vents['converged'] else 'does not converge'} at "
        f"{vents['outer']:,} outer iterations, {ratio:.3f} times the default's "
        f"{default['outer']:,}, against a limit of {RATIO_LIMIT}."
    )


def box(results: Path) -> str:
    """Prediction (c): the closed box's signed sum at every outer iteration."""
    r = load(results / "builder61" / "closed_box.json")
    sums = [abs(x) for x in r["signed_sums"]]
    floor = 1e-13 * r["flux_scale"]
    rows = [
        [
            f"{r['stop']} at {r['outer']:,}",
            f"{r['flux_scale']:.4g}",
            f"{sums[0]:.2e}",
            f"{max(sums):.2e}",
            f"{max(sums) / r['flux_scale']:.1e}",
            "yes" if max(sums) <= floor else "NO",
        ]
    ]
    return table(
        [
            "Stop",
            "Flux scale F (kg/s per m)",
            "Signed sum, outer iteration 1 (kg/s per m)",
            "Worst over every outer iteration",
            "Worst over F",
            "At rounding (at most 1e-13 F) from the first",
        ],
        rows,
    )


def march(results: Path, room: str, source: str) -> dict:
    """One march record."""
    return load(results / "builder61" / f"transport_{room}_{source}_sc{SC}.json")


def terms(rec: dict, k: str) -> dict:
    """One march's contribution to each metric for class k."""
    per = rec["per_class"][k]
    emission = float(rec["source"]["emission"])
    return {
        "M1": sum(per["by_surface"].get(s, 0.0) for s in PROCESS_SURFACES),
        "M2": sum(per["sensors"][s] / emission for s in OPERATOR_SENSORS),
        "M3": float(per["net_removal_over_source"]),
        "stop": per["stop"],
        "diameter": per["diameter"],
    }


def metrics(results: Path) -> str:
    """The three Phase 3 metrics, default against vented, per class."""
    rows = []
    for k in CLASSES:
        values = {}
        stops = {}
        diameter = None
        for name, room in ROOMS:
            t = [terms(march(results, name, s), k) for s in SOURCES]
            diameter = t[0]["diameter"]
            values[room] = {
                "M1": sum(x["M1"] for x in t),
                "M2": sum(x["M2"] for x in t),
                "M3": sum(x["M3"] for x in t) / len(t),
            }
            stops[room] = ", ".join(x["stop"] for x in t)
        for metric, label in (
            ("M1", "M1 process-top deposition (1/s per m, lower better)"),
            ("M2", "M2 operator exposure (s/m^3 summed, lower better)"),
            ("M3", "M3 removal, net (higher better)"),
        ):
            d, v = values["default"][metric], values["vents"][metric]
            change = (v - d) / d if d != 0.0 else float("nan")
            rows.append(
                [
                    f"{diameter * 1e6:g}",
                    label,
                    f"{d:.4g}",
                    f"{v:.4g}",
                    f"{change:+.1%}",
                    f"{stops['default']} / {stops['vents']}",
                ]
            )
    return table(
        [
            "Class (um)",
            "Metric",
            "Default",
            "Vents",
            "Vents against default",
            "March stops S1, S2, S3 (default / vents)",
        ],
        rows,
    )


def sensors(results: Path) -> str:
    """Prediction (d): near_door and above_gap_1, each source and class."""
    rows = []
    for k in CLASSES:
        for source in SOURCES:
            d = march(results, "default_320x120", source)["per_class"][k]
            v = march(results, "vents_320x120", source)["per_class"][k]
            cells = []
            for sensor in PREDICTION_D_SENSORS:
                a, b = d["sensors"][sensor], v["sensors"][sensor]
                # Step 8's floor (transport47.FLOOR_RATIO of the source cell): a
                # reading below it is no particle reaching the sensor.
                seen = [
                    "yes" if sensor in room["sensors_above_floor"] else "no"
                    for room in (d, v)
                ]
                cells += [
                    f"{a:.3e}",
                    f"{b:.3e}",
                    "raised" if b > a else "lowered",
                    " / ".join(seen),
                ]
            rows.append([f"{d['diameter'] * 1e6:g}", source, *cells])
    return table(
        [
            "Class (um)",
            "Source",
            "near_door default",
            "near_door vents",
            "near_door",
            "near_door above the floor (default / vents)",
            "above_gap_1 default",
            "above_gap_1 vents",
            "above_gap_1",
            "above_gap_1 above the floor (default / vents)",
        ],
        rows,
    )


def removal(results: Path) -> str:
    """The vented room's removal paths, per source and class: gross outflow, inflow, net."""
    rows = []
    for k in CLASSES:
        for source in SOURCES:
            for name, room in ROOMS:
                per = march(results, name, source)["per_class"][k]
                rows.append(
                    [
                        f"{per['diameter'] * 1e6:g}",
                        source,
                        room,
                        f"{per['outflow_rate_over_source']:.4f}",
                        f"{per['inflow_rate_over_source']:.4f}",
                        f"{per['net_removal_over_source']:.4f}",
                        f"{per['budget_deposition_rate_over_source']:.4f}",
                        per["stop"],
                        f"{per['t_end']:g}",
                    ]
                )
    return table(
        [
            "Class (um)",
            "Source",
            "Room",
            "Outflow over Q",
            "Inflow over Q",
            "Net removal over Q (M3 term)",
            "Deposition over Q",
            "March stop",
            "Simulated s",
        ],
        rows,
    )


def plants(results: Path) -> str:
    """The planted defects, each with its control and its result."""
    rows = []
    for p in load(results / "builder61" / "plants.json"):
        rows.append(
            [
                p["plant"],
                p["file"],
                ", ".join(t.removeprefix("tests/") for t in p.get("tests", [])),
                p.get("planted", p.get("error", "")),
                "caught" if p.get("caught") else "SURVIVED",
            ]
        )
    return table(["Planted defect", "File", "Tests", "Planted run", "Result"], rows)


def render(results: Path) -> dict[str, str]:
    """Every table, by its marker name."""
    return {
        "hashes": hashes(results),
        "flows": flows(results) + "\n\n" + verdict_b(results),
        "box": box(results),
        "metrics": metrics(results),
        "sensors": sensors(results),
        "removal": removal(results),
        "plants": plants(results),
    }


def splice(text: str, tables: dict[str, str]) -> str:
    """The report with each marked table replaced."""
    for name, body in tables.items():
        pattern = re.compile(
            rf"(<!-- BEGIN tables61: {name} -->\n).*?(<!-- END tables61: {name} -->)",
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
    results = hash61.main_results().parent
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
