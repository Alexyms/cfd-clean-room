"""The tables of prompt 82, the placement search's scaling, from its records.

    python docs/reports/probe82/tables82.py            # write the tables, splice the report
    python docs/reports/probe82/tables82.py --check    # regenerate and compare

Reads ``results/builder82/`` (untracked, this machine): ``tables.json`` and
``tables/<name>.npz``, written by ``probe82.py build``, and one record per
case in ``cases/``, written by ``probe82.py queue``. Writes ``tables82.md``
beside it and copies each table into ``docs/reports/search_scaling.md``
between ``<!-- tables82 NAME begin/end -->``. ``--check`` regenerates both
and compares.
"""

import argparse
import json
import math
import sys
from itertools import pairwise
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import probe82  # noqa: E402 -- follows sys.path.insert

REPORT = HERE.parent / "search_scaling.md"
OUT = HERE / "tables82.md"
MB = 2**20
# The prompt's figures: the other scenarios' units and a position's.
OTHER_UNITS = 15
UNITS_PER_POSITION = 5
# Prediction (a)'s units, (b)'s masks, (c)'s minutes, (d)'s share.
A_UNITS = 26
B_MASKS = 2000
C_SECONDS = 5 * 60
D_SHARE = 0.10
GRIDS = {"320x120": "320x120", "160x60": "160x60", "noise": "noise (bound)"}


def tables_meta() -> dict[str, dict]:
    """The tables' summaries, by name."""
    meta = json.loads((probe82.RESULTS / "tables.json").read_text(encoding="utf-8"))
    return {t["name"]: t for t in meta["tables"]}


def records() -> dict[tuple[str, int, str], dict]:
    """Every case's record, by (table, count, method)."""
    out = {}
    for method, name, count in probe82.cases():
        path = probe82.case_path(method, name, count)
        if path.exists():
            out[(name, count, method)] = json.loads(path.read_text(encoding="utf-8"))
    return out


def realistic(name: str) -> bool:
    """A table of the prompt's or the edge's, not the noise bound."""
    return not name.startswith("noise")


def by_units(names: list[str], meta: dict) -> list[str]:
    """Table names ordered by grid, then units."""
    order = {"320x120": 0, "160x60": 1, "noise": 2}
    return sorted(names, key=lambda n: (order[meta[n]["grid"]], meta[n]["units"]))


def seconds(value: float | None) -> str:
    """A time in seconds, three significant figures."""
    if value is None:
        return "-"
    return f"{value:.3g}" if value < 1000 else f"{value:,.0f}"


def mbytes(value: float | None) -> str:
    """Bytes as MB."""
    return "-" if value is None else f"{value / MB:,.0f}"


def latency(value: float | None) -> str:
    """A latency in seconds, four decimals, or none."""
    if value is None:
        return "no cover"
    return "inf" if math.isinf(value) else f"{value:.4f}"


def floor_of(name: str) -> float:
    """The largest over the units of the least latency any candidate gives.

    A lower bound on every layout's worst case, of any count.
    """
    table, _ = probe82.load_table(name)
    return float(np.max(np.min(table, axis=0)))


def runs_text(labels: list[list[str]]) -> str:
    """The table's units grouped by run, in order: run (units)."""
    groups: list[list] = []
    for scenario, _, _ in labels:
        if groups and groups[-1][0] == scenario:
            groups[-1][1] += 1
        else:
            groups.append([scenario, 1])
    if groups[0][0] == "noise":
        return f"independent latencies ({groups[0][1]} units)"
    return ", ".join(f"{s.removesuffix('_160x60')} ({n})" for s, n in groups)


# ---------------------------------------------------------------------------
# The tables
# ---------------------------------------------------------------------------


def inputs_table(meta: dict) -> list[str]:
    """I1: the tables searched."""
    lines = [
        "### I1. The unit tables: the product search's candidates (the product file's mount, standoff and stand band) by the units, built with main's code on cumulative maps; the floor is the largest over the units of the least latency any candidate gives, a lower bound on every layout's worst case",
        "",
        "| Table | Grid | Maps | Candidates | Units | Units, by run in order | Floor (s) |",
        "|---|---|---|---|---|---|---|",
    ]
    maps = {
        "320x120": "step (prompt 81's /arrival/)",
        "160x60": "frames (prompt 79's sweep, on the channels)",
        "noise": "none: uniform on [0, 30] s, half unreached, seed 82",
    }
    for name in by_units(list(meta), meta):
        t = meta[name]
        lines.append(
            f"| {name} | {GRIDS[t['grid']]} | {maps[t['grid']]} | {t['candidates']:,} | "
            f"{t['units']} | {runs_text(t['labels'])} | {latency(floor_of(name))} |"
        )
    return lines


def cost_table(meta: dict, recs: dict) -> list[str]:
    """C1: every case's cost."""
    lines = [
        "### C1. Every case: its status, wall and CPU time, and the process's private bytes before the method and at their peak (one process per case, one at a time; a case estimated above 8 GB is skipped, one past 30 minutes stopped)",
        "",
        "| Table | Units | Sensors | Method | Status | Wall (s) | CPU (s) | Private before (MB) | Peak private (MB) | Peak above before (MB) | Estimate (MB) |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name in by_units(list(meta), meta):
        for count in probe82.COUNTS:
            for method in probe82.METHODS:
                r = recs.get((name, count, method))
                if r is None:
                    continue
                ok = r["status"] == "ok"
                above = r["peak_private"] - r["private_before"] if ok else None
                lines.append(
                    f"| {name} | {r['units']} | {count} | {method} | {r['status']} | "
                    f"{seconds(r.get('wall'))} | {seconds(r.get('cpu'))} | "
                    f"{mbytes(r.get('private_before'))} | {mbytes(r.get('peak_private'))} | "
                    f"{mbytes(above)} | {mbytes(r['estimate'])} |"
                )
    return lines


def cover_curve(meta: dict, recs: dict) -> tuple[list[str], dict]:
    """C2: the built cover at four sensors against the units, and its growth."""
    lines = [
        "### C2. The built set cover against the units on the realistic tables, at each count: wall time and peak private bytes above the start, and the factor per unit between consecutive sizes measured",
        "",
        "| Table | Units | Sensors | Status | Wall (s) | Peak above before (MB) | Estimate (MB) | Wall factor per unit | Memory factor per unit |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    fit: dict = {}
    names = [n for n in by_units(list(meta), meta) if realistic(n)]
    for count in (2, 3, 4):
        last = None
        points = []
        skipped = False
        for name in names:
            r = recs.get((name, count, "cover"))
            if r is None or skipped:
                continue
            # The sizes past the first one skipped are skipped too; one row says so.
            skipped = r["status"] == "skipped"
            u = r["units"]
            wall_f = mem_f = "-"
            if r["status"] == "ok":
                above = r["peak_private"] - r["private_before"]
                if meta[name]["grid"] == "160x60":
                    if last is not None:
                        step = u - last[0]
                        wall_f = f"{(r['wall'] / last[1]) ** (1 / step):.2f}"
                        mem_f = f"{(above / last[2]) ** (1 / step):.2f}"
                    last = (u, r["wall"], above)
                    points.append((u, r["wall"], above))
            else:
                above = None
            lines.append(
                f"| {name} | {u} | {count} | {r['status']} | {seconds(r.get('wall'))} | "
                f"{mbytes(above)} | {mbytes(r['estimate'])} | {wall_f} | {mem_f} |"
            )
        fit[count] = points
    return lines, fit


def extrapolation_table(fit: dict) -> tuple[list[str], dict]:
    """C3: the built cover at four sensors carried past the measured sizes."""
    points = fit[4]
    us = np.array([p[0] for p in points], dtype=float)
    wall = np.log2([p[1] for p in points])
    mem = np.log2([p[2] for p in points])
    top = points[-1]
    sw, iw = np.polyfit(us, wall, 1)
    sm, im = np.polyfit(us, mem, 1)
    lines = [
        f"### C3. Extrapolated, not measured: the built set cover at four sensors past {top[0]} units, from a least-squares line of log2 wall and log2 peak above the start against the units over the 160x60 sizes measured ({', '.join(str(int(u)) for u in us)}): {2**sw:.2f} times the time and {2**sm:.2f} times the memory per unit",
        "",
        "| Units | Release positions (15 + 5 each) | Wall, extrapolated (s) | Peak above the start, extrapolated (MB) | Within 30 minutes | Within 8 GB |",
        "|---|---|---|---|---|---|",
    ]
    out = {}
    for u in (28, 30, 35):
        w = 2 ** (sw * u + iw)
        m = 2 ** (sm * u + im)
        out[u] = (w, m)
        positions = (u - OTHER_UNITS) / UNITS_PER_POSITION
        lines.append(
            f"| {u} | {positions:g} | {seconds(w)} | {mbytes(m)} | "
            f"{'yes' if w <= probe82.TIME_LIMIT else 'no'} | "
            f"{'yes' if m <= probe82.MEMORY_LIMIT else 'no'} |"
        )
    return lines, {"wall_factor": 2**sw, "memory_factor": 2**sm, "rows": out}


def masks_table(meta: dict, recs: dict) -> tuple[list[str], dict]:
    """M1: the masks at the optimum and at the largest latency, and the enumeration's work."""
    lines = [
        "### M1. The masks (each candidate's set of units detected within a threshold), distinct and maximal, at the worst-case optimum's threshold and at the largest finite latency; the enumeration's work over its bisection (feasibility nodes, pairs of masks tested, covers enumerated at the optimum); and the combinations of at most the count of the optimum's maximal masks a literal enumeration would try",
        "",
        "| Table | Units | Sensors | Optimum (s) | Floor (s) | Optimum is the floor | Distinct at the optimum | Maximal at the optimum | Distinct at the largest | Maximal at the largest | Bisection steps | Nodes | Pairs tested | Covers at the optimum | Literal combinations |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    largest_maximal: dict[str, int] = {}
    for name in by_units(list(meta), meta):
        floor = floor_of(name)
        for count in probe82.COUNTS:
            r = recs.get((name, count, "enum"))
            if r is None:
                continue
            if r["status"] != "ok":
                lines.append(
                    f"| {name} | {r['units']} | {count} | {r['status']} | {latency(floor)} | - | - | - | - | - | - | - | - | - | - |"
                )
                continue
            s = r["stats"]
            opt, top = r["at_optimum"], r["at_largest"]
            if opt is not None:
                largest_maximal[name] = max(
                    largest_maximal.get(name, 0), opt["maximal"]
                )
            literal = (
                f"{probe82.literal_combinations(opt['maximal'], count):,}"
                if opt is not None
                else "-"
            )
            lines.append(
                f"| {name} | {r['units']} | {count} | {latency(r['worst'])} | {latency(floor)} | "
                f"{'-' if r['worst'] is None else ('yes' if r['worst'] == floor else 'no')} | "
                f"{'-' if opt is None else opt['distinct']} | {'-' if opt is None else opt['maximal']} | "
                f"{top['distinct']} | {top['maximal']} | {s['bisections']} | {s['nodes']:,} | "
                f"{s['pairs']:,} | {s.get('covers', '-')} | {literal} |"
            )
    return lines, largest_maximal


def structure_table(meta: dict, recs: dict) -> tuple[list[str], dict]:
    """M2: what the maximal masks are on the realistic tables, at four sensors' optimum."""
    lines = [
        "### M2. The realistic tables' masks at the four-sensor optimum and at the largest finite latency: the runs whose channels' masks are nested over every candidate (each candidate's channels detected within the threshold are a chain under inclusion), and each maximal mask's units, by run",
        "",
        "| Table | Threshold | At (s) | Runs whose masks nest | Maximal masks | Pairwise disjoint | Units of each maximal mask, by run |",
        "|---|---|---|---|---|---|---|",
    ]
    out = {"nested": 0, "runs": 0, "maximal": set()}
    for name in by_units([n for n in meta if realistic(n)], meta):
        r = recs.get((name, 4, "enum"))
        if r is None or r["status"] != "ok":
            continue
        table, _ = probe82.load_table(name)
        labels = meta[name]["labels"]
        runs: dict[str, list[int]] = {}
        for u, (scenario, _, _) in enumerate(labels):
            runs.setdefault(scenario, []).append(u)
        largest = float(table[np.isfinite(table)].max())
        for title, threshold in (("optimum", r["worst"]), ("largest", largest)):
            hit = table <= threshold
            nested = 0
            for units in runs.values():
                rows = sorted({tuple(row) for row in hit[:, units]}, key=sum)
                nested += all(
                    all(a <= b for a, b in zip(x, y, strict=True))
                    for x, y in pairwise(rows)
                )
            maximal = probe82.placement._maximal(
                np.unique(probe82.masks_at(table, threshold))
            )
            parts = []
            for mask in maximal:
                held = [
                    f"{s.removesuffix('_160x60')} {sum(int(mask) >> u & 1 for u in us)}"
                    for s, us in runs.items()
                    if any(int(mask) >> u & 1 for u in us)
                ]
                parts.append("{" + ", ".join(held) + "}")
            ints = [int(m) for m in maximal]
            disjoint = all(
                a & b == 0 for i, a in enumerate(ints) for b in ints[i + 1 :]
            )
            out["nested"] += nested
            out["runs"] += len(runs)
            out["maximal"].add(int(maximal.size))
            lines.append(
                f"| {name} | {title} | {latency(threshold)} | {nested} of {len(runs)} | "
                f"{maximal.size} | {'yes' if disjoint else 'no'} | {'; '.join(parts)} |"
            )
    return lines, out


def agreement_table(meta: dict, recs: dict) -> tuple[list[str], dict]:
    """A1: the enumeration against the built cover and the exhaustive search."""
    lines = [
        "### A1. The enumeration against the built set cover (two to four sensors) and the exhaustive search `search` takes at one and two (`_best_one`, `_best_two` under the worst-case rule), wherever both ran: the worst case, and the layout (the same cells; no cover where no layout of the count detects every unit, the worst case inf for both)",
        "",
        "| Table | Units | Sensors | Against | Enumeration's worst (s) | Its worst (s) | Same worst case | Same cells | Enumeration's mean (s) | Its mean (s) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    result = {"pairs": 0, "worst_differs": [], "cells_differ": [], "mean_differs": []}
    for name in by_units(list(meta), meta):
        for count in probe82.COUNTS:
            e = recs.get((name, count, "enum"))
            if e is None or e["status"] != "ok":
                continue
            for other in ("cover", "exhaustive"):
                o = recs.get((name, count, other))
                if o is None or o["status"] != "ok":
                    continue
                ew = math.inf if e["worst"] is None else e["worst"]
                ow = o["worst"]
                same_worst = ew == ow
                # No layout of the count detects every unit: the enumeration, as
                # the built cover, returns no cells, and the exhaustive search the
                # best by the mean among layouts that all have an infinite worst case.
                none = math.isinf(ew) and math.isinf(ow)
                same_cells = e["columns"] == o["columns"]
                result["pairs"] += 1
                if not same_worst:
                    result["worst_differs"].append((name, count, other))
                elif other == "cover" and not same_cells and e["mean"] != o["mean"]:
                    result["mean_differs"].append((name, count, other))
                if not same_cells and not none:
                    result["cells_differ"].append((name, count, other))
                cells = "no cover" if none else ("yes" if same_cells else "no")
                mean_e = "-" if e["mean"] is None else f"{e['mean']:.4f}"
                lines.append(
                    f"| {name} | {e['units']} | {count} | {other} | {latency(ew)} | "
                    f"{latency(ow)} | {'yes' if same_worst else 'no'} | "
                    f"{cells} | {mean_e} | {o['mean']:.4f} |"
                )
    lines += [
        "",
        f"Compared: {result['pairs']}. A different worst case: "
        f"{len(result['worst_differs'])}. Different cells where a cover exists: "
        f"{len(result['cells_differ'])}.",
    ]
    return lines, result


def greedy_table(meta: dict, recs: dict) -> tuple[list[str], dict]:
    """G1: the greedy mean search against the exact worst-case optimum."""
    lines = [
        "### G1. The mean rule's greedy search (`_greedy`: additions, then one-sensor exchanges) against the exact worst-case optimum (the enumeration's), two to four sensors; `search` itself runs the greedy only above two, and at two takes every pair",
        "",
        "| Table | Units | Sensors | Exact worst (s) | Greedy worst (s) | Greedy's worst over the exact | Exact optimum's mean (s) | Greedy mean (s) | Greedy wall (s) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    gaps: dict[tuple[str, int], float] = {}
    for name in by_units(list(meta), meta):
        for count in (2, 3, 4):
            e = recs.get((name, count, "enum"))
            g = recs.get((name, count, "greedy"))
            if e is None or g is None or e["status"] != "ok" or g["status"] != "ok":
                continue
            if e["worst"] is None:
                continue
            gap = g["worst"] / e["worst"] - 1.0
            gaps[(name, count)] = gap
            lines.append(
                f"| {name} | {e['units']} | {count} | {latency(e['worst'])} | "
                f"{latency(g['worst'])} | {gap:+.1%} | {e['mean']:.4f} | "
                f"{g['mean']:.4f} | {seconds(g['wall'])} |"
            )
    return lines, gaps


def largest_ok(meta: dict, recs: dict, method: str, kinds: tuple[str, ...]) -> tuple:
    """The most units at four sensors a method finished on, and the first case it did not."""
    best, first_not = None, None
    for name in by_units([n for n in meta if meta[n]["grid"] in kinds], meta):
        r = recs.get((name, 4, method))
        if r is None:
            continue
        if r["status"] == "ok":
            if best is None or r["units"] > best[0]:
                best = (r["units"], r["wall"], r["peak_private"] - r["private_before"])
        elif first_not is None or r["units"] < first_not[0]:
            first_not = (r["units"], r["status"], r["estimate"])
    return best, first_not


def positions(units: int) -> int:
    """Dust release positions a table of UNITS carries beside the other fifteen."""
    return max(0, (units - OTHER_UNITS) // UNITS_PER_POSITION)


def positions_table(meta: dict, recs: dict, extra: dict) -> tuple[list[str], dict]:
    """N1: how many dust release positions each method carried at four sensors."""
    lines = [
        "### N1. At four sensors, the most units each method finished on within the limits, and the dust release positions that carries at five units a position on top of the other scenarios' fifteen (20 units: one position, 25: two, 60: nine)",
        "",
        "| Method | Tables | Most units finished | Its wall (s) | Its peak above the start (MB) | First case not finished | Positions carried | What stops it next |",
        "|---|---|---|---|---|---|---|---|",
    ]
    out = {}
    largest = max(meta[n]["units"] for n in meta if realistic(n))
    for method in ("cover", "enum", "greedy"):
        for kinds, title in (
            (("320x120", "160x60"), "realistic"),
            (("noise",), "noise (bound)"),
        ):
            best, first_not = largest_ok(meta, recs, method, kinds)
            if best is None and first_not is None:
                continue
            if first_not is not None:
                why = {
                    "skipped": f"its estimate, {mbytes(first_not[2])} MB, is above 8 GB",
                    "over": "it ran past 30 minutes",
                    "error": "it raised",
                }[first_not[1]]
                nxt = f"{first_not[0]} units: {first_not[1]} ({why})"
                stop = why
            else:
                nxt = "none"
                stop = (
                    f"not reached: the largest table measured is {largest} units; the "
                    f"masks are int64, so the built code's masks hold at most "
                    f"{probe82.PROBE_CAP} units"
                    if method != "greedy"
                    else f"not reached: the largest table measured is {largest} units"
                )
            if method == "cover" and first_not is not None and title == "realistic":
                w, m = extra["rows"][30]
                stop += (
                    f"; 30 units (three positions) extrapolate to {seconds(w)} s and "
                    f"{mbytes(m)} MB (C3)"
                )
            carried = positions(best[0]) if best is not None else 0
            out[(method, title)] = (best, first_not, carried)
            lines.append(
                f"| {method} | {title} | {'-' if best is None else best[0]} | "
                f"{'-' if best is None else seconds(best[1])} | "
                f"{'-' if best is None else mbytes(best[2])} | {nxt} | {carried} | {stop} |"
            )
    return lines, out


def predictions_table(v: dict) -> list[str]:
    """P1: the predictions against the measurements."""
    lines = [
        "### P1. The predictions against the measurements",
        "",
        "| Prediction | As committed | Measured | Verdict |",
        "|---|---|---|---|",
    ]
    for key, text, measured, holds in v["rows"]:
        lines.append(
            f"| ({key}) | {text} | {measured} | {'holds' if holds else 'FAILS'} |"
        )
    return lines


def verdicts(
    meta: dict, recs: dict, largest_maximal: dict, agree: dict, gaps: dict, n1: dict
) -> dict:
    """The four predictions, judged on the realistic tables as the prompt states them."""
    rows = []
    best, first_not = n1[("cover", "realistic")][:2]
    limited = (
        {"skipped": "memory", "over": "time"}.get(first_not[1]) if first_not else None
    )
    a_holds = limited == "time" and best[0] <= A_UNITS
    rows.append(
        (
            "a",
            "The built set cover is limited by time before memory: on this laptop it finishes within 30 minutes up to no more than 26 units at four sensors",
            f"N1, C2: at four sensors it finished at {best[0]} units in {seconds(best[1])} s "
            f"({mbytes(best[2])} MB above the start); the first size not run, "
            f"{first_not[0]} units, was stopped by {limited} (estimate "
            f"{mbytes(first_not[2])} MB)",
            a_holds,
        )
    )
    real = {n: m for n, m in largest_maximal.items() if realistic(n)}
    noise = {n: m for n, m in largest_maximal.items() if not realistic(n)}
    top = max(real.values())
    rows.append(
        (
            "b",
            "The maximal masks at the optimum's threshold stay below 2,000 up to 60 units",
            f"M1: at most {top} on the realistic tables (up to "
            f"{max(meta[n]['units'] for n in real)} units); the noise bound, not a "
            f"realistic table, reaches {max(noise.values()) if noise else '-'}",
            top < B_MASKS,
        )
    )
    e60 = [recs.get(("coarse60", c, "enum")) for c in probe82.COUNTS]
    slowest = max((r["wall"] for r in e60 if r and r["status"] == "ok"), default=None)
    all_ok = all(r and r["status"] == "ok" for r in e60)
    n60 = [recs.get(("noise60", c, "enum")) for c in probe82.COUNTS]
    n60_text = "; ".join(
        f"at {c} {'sensor' if c == 1 else 'sensors'} {r['status']}"
        + (f" in {seconds(r['wall'])} s" if r["status"] == "ok" else "")
        for c, r in zip(probe82.COUNTS, n60, strict=True)
        if r is not None
    )
    c_holds = (
        not agree["worst_differs"]
        and all_ok
        and slowest is not None
        and slowest <= C_SECONDS
    )
    rows.append(
        (
            "c",
            "The enumeration of combinations of at most four maximal masks returns the built cover's worst case wherever both run, and finishes within 5 minutes at 60 units",
            f"A1: the same worst case in {agree['pairs'] - len(agree['worst_differs'])} of "
            f"{agree['pairs']} comparisons; on the 60-unit table the slowest count took "
            f"{seconds(slowest)} s; on the noise bound at 60 units: {n60_text or '-'}",
            c_holds,
        )
    )
    real_gaps = {k: g for k, g in gaps.items() if realistic(k[0])}
    over = sorted((k for k, g in real_gaps.items() if g > D_SHARE), key=str)
    worst_gap = max(real_gaps.values())
    rows.append(
        (
            "d",
            "Where the exact worst-case optimum is known, the greedy mean search's worst case is within 10% of it",
            f"G1: largest gap {worst_gap:+.1%} on the realistic tables; above 10% in "
            f"{len(over)} of {len(real_gaps)} cases, all at "
            f"{', '.join(sorted({str(c) for _, c in over})) or '-'} sensors; at three and "
            f"four the largest gap is "
            f"{max(g for (n, c), g in real_gaps.items() if c > 2):+.1%}",
            not over,
        )
    )
    return {"rows": rows}


def stops_table(agree: dict) -> list[str]:
    """X1: the prompt's stop condition."""
    differs = agree["worst_differs"]
    return [
        "### X1. The prompt's stop condition",
        "",
        "| Condition | Met |",
        "|---|---|",
        "| The enumeration and the built cover (or the exhaustive search) disagree on a worst case anywhere (A1) | "
        + (
            "yes: " + ", ".join(f"{n} at {c} ({o})" for n, c, o in differs)
            if differs
            else "no"
        )
        + " |",
    ]


def build() -> dict[str, list[str]]:
    """Every table, by its marker name."""
    meta = tables_meta()
    recs = records()
    blocks: dict[str, list[str]] = {}
    blocks["inputs"] = inputs_table(meta)
    blocks["cost"] = cost_table(meta, recs)
    blocks["curve"], fit = cover_curve(meta, recs)
    blocks["extrapolation"], extra = extrapolation_table(fit)
    blocks["masks"], largest_maximal = masks_table(meta, recs)
    blocks["structure"], _ = structure_table(meta, recs)
    blocks["agreement"], agree = agreement_table(meta, recs)
    blocks["greedy"], gaps = greedy_table(meta, recs)
    blocks["positions"], n1 = positions_table(meta, recs, extra)
    v = verdicts(meta, recs, largest_maximal, agree, gaps, n1)
    blocks["predictions"] = predictions_table(v)
    blocks["stops"] = stops_table(agree)
    return blocks


ORDER = (
    "predictions",
    "stops",
    "inputs",
    "cost",
    "curve",
    "extrapolation",
    "masks",
    "structure",
    "agreement",
    "greedy",
    "positions",
)


def text_of(blocks: dict[str, list[str]]) -> str:
    """tables82.md."""
    parts = ["# Prompt 82's tables (generated by tables82.py; do not edit)", ""]
    for name in ORDER:
        parts += [*blocks[name], ""]
    return "\n".join(parts)


def splice(text: str, blocks: dict[str, list[str]]) -> str:
    """The report with each marked block replaced."""
    for name, lines in blocks.items():
        begin, end = f"<!-- tables82 {name} begin -->", f"<!-- tables82 {name} end -->"
        if begin not in text:
            continue
        head, rest = text.split(begin, 1)
        _, tail = rest.split(end, 1)
        text = head + begin + "\n" + "\n".join(lines) + "\n" + end + tail
    return text


def main() -> int:
    """Write or check the tables."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    blocks = build()
    tables = text_of(blocks)
    report = REPORT.read_text(encoding="utf-8")
    spliced = splice(report, blocks)
    if args.check:
        ok = True
        if OUT.read_text(encoding="utf-8") != tables:
            print(f"{OUT.name} differs from the records")
            ok = False
        if report != spliced:
            print(f"{REPORT.name}'s copies differ from the records")
            ok = False
        print("match" if ok else "differ")
        return 0 if ok else 1
    OUT.write_text(tables, encoding="utf-8")
    REPORT.write_text(spliced, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
