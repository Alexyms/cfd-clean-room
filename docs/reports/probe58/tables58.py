"""The Phase 3 gate report's per-metric tables and figure from step 8's records.

Usage:
    python tables58.py tables     print the metric, ranking, margin and summary tables and
                                  write tables58.md and tables58.json beside this file
    python tables58.py figure     draw docs/reports/phase3_metrics.png
    python tables58.py all        both
    python tables58.py --check    regenerate the tables and compare them with the committed
                                  tables58.md and tables58.json; exit 1 on any difference

The metrics are metrics58.md's, defined before this file existed: M1 the
process-top deposition (step 8's score), M2 the operator exposure at the two
operator-height sensors over each source's emission, M3 the removal through
the returns and the hood. The records are results/builder56/transport_*.json
(untracked), read where they are; the starred L2 rows are step 8's labelled
supplement (tables56.SUPPLEMENT). The ranking and margin rule is step 8
section 5.4's, with a neighbouring difference that does not exceed every
applicable change reported as a tie.
"""

import argparse
import itertools
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RECORDS = ROOT / "results" / "builder56"
TABLES_MD = HERE / "tables58.md"
TABLES_JSON = HERE / "tables58.json"
FIGURE = ROOT / "docs" / "reports" / "phase3_metrics.png"

LAYOUTS = ("L0", "L1", "L2")
GRIDS = ("160x60", "320x120")
PRODUCT_GRID = "320x120"
SOURCES = ("S1", "S2", "S3")
CLASSES = ("2", "4")
SCHMIDTS = (0.2, 0.7, 1.3)
BASE_SC = 0.7
PROCESS_SURFACES = ("litho_tool top", "etch_chamber top")
OPERATOR_SENSORS = ("near_door", "hood_entry")
# Step 8's supplementary rows (tables56.SUPPLEMENT, restated so this file reads
# the records on its own): L2's marches run on the labelled cap-raised rerun on
# 320x120 and on the field at the cap on 160x60; both are marked L2*.
SUPPLEMENT: dict[tuple[str, str], tuple[str, str]] = {
    ("L2", "320x120"): ("L2_320x120_cap12000", ""),
    ("L2", "160x60"): ("L2_160x60", "_cap"),
}
LEGEND = (
    "L2* is supplementary (step 8 section 5.2): on 320x120 the rerun with the cap raised, "
    "which stops by the rule; on 160x60 the field at the cap, a cycle the rule does not stop on."
)
METRICS = {
    "M1": {"name": "process-top deposition", "better": "lower", "combine": "sum"},
    "M2": {"name": "operator exposure", "better": "lower", "combine": "sum"},
    "M3": {"name": "removal", "better": "higher", "combine": "mean"},
}
CASES = [(GRIDS[0], BASE_SC), *((PRODUCT_GRID, sc) for sc in SCHMIDTS)]
TIE = " ~ "
ORDER = " > "


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


def sc_key(sc: float) -> str:
    return f"{sc:g}"


def label_of(layout: str, grid: str) -> str:
    return layout + ("*" if (layout, grid) in SUPPLEMENT else "")


def record_name(layout: str, grid: str, source: str, sc: float) -> str:
    flow, suffix = SUPPLEMENT.get((layout, grid), (f"{layout}_{grid}", ""))
    return f"transport_{flow}_{source}_sc{sc_key(sc)}{suffix}"


def load(name: str) -> dict:
    path = RECORDS / f"{name}.json"
    if not path.exists():
        sys.exit(f"record missing: {path}")
    return json.loads(path.read_text())


def metric_terms(rec: dict, k: str) -> dict:
    """One march's contribution to each metric, per class, with its stop."""
    per = rec["per_class"][k]
    emission = float(rec["source"]["emission"])
    for field in ("by_surface", "sensors", "outflow_rate_over_source", "stop"):
        if field not in per:
            sys.exit(f"{rec['name']}: class {k} lacks {field}")
    for sensor in OPERATOR_SENSORS:
        if sensor not in per["sensors"]:
            sys.exit(f"{rec['name']}: class {k} sensors lack {sensor}")
    m1 = sum(per["by_surface"].get(s, 0.0) for s in PROCESS_SURFACES)
    exposure = {s: per["sensors"][s] / emission for s in OPERATOR_SENSORS}
    return {
        "M1": m1,
        "M1_terms": {s: per["by_surface"].get(s, 0.0) for s in PROCESS_SURFACES},
        "M2": sum(exposure.values()),
        "M2_terms": exposure,
        "M3": float(per["outflow_rate_over_source"]),
        "stop": per["stop"],
        "diameter": per["diameter"],
    }


def values() -> dict:
    """Every (layout, grid, Sc_t, class) the records allow, keyed LAYOUT/GRID/SC/CLASS."""
    out: dict[str, dict] = {}
    for layout, (grid, sc), k in itertools.product(LAYOUTS, CASES, CLASSES):
        names = [record_name(layout, grid, source, sc) for source in SOURCES]
        recs = [load(n) for n in names]
        terms = {
            source: metric_terms(rec, k)
            for source, rec in zip(SOURCES, recs, strict=True)
        }
        entry = {
            "layout": layout,
            "label": label_of(layout, grid),
            "supplementary": (layout, grid) in SUPPLEMENT,
            "grid": grid,
            "sc": sc,
            "class": k,
            "diameter": terms[SOURCES[0]]["diameter"],
            "records": names,
            "stops": {source: terms[source]["stop"] for source in SOURCES},
            "per_source": terms,
            "M1": sum(terms[s]["M1"] for s in SOURCES),
            "M2": sum(terms[s]["M2"] for s in SOURCES),
            "M3": sum(terms[s]["M3"] for s in SOURCES) / len(SOURCES),
        }
        out[key(layout, grid, sc, k)] = entry
    return out


def key(layout: str, grid: str, sc: float, k: str) -> str:
    return f"{layout}/{grid}/{sc_key(sc)}/{k}"


# ---------------------------------------------------------------------------
# Ranking and margins
# ---------------------------------------------------------------------------


def changes(v: dict, layout: str, k: str, metric: str) -> dict:
    """The grid's and the Schmidt number's change to one layout's metric (metrics58.md)."""
    coarse = v[key(layout, GRIDS[0], BASE_SC, k)][metric]
    fine = v[key(layout, PRODUCT_GRID, BASE_SC, k)][metric]
    sc_changes = [
        abs(v[key(layout, PRODUCT_GRID, sc, k)][metric] - fine)
        for sc in SCHMIDTS
        if sc != BASE_SC
    ]
    return {"grid": abs(fine - coarse), "schmidt": max(sc_changes)}


def order_for(v: dict, grid: str, sc: float, k: str, metric: str) -> list[str]:
    """Layouts best first on one case and metric."""
    reverse = METRICS[metric]["better"] == "higher"
    return sorted(
        LAYOUTS, key=lambda lay: v[key(lay, grid, sc, k)][metric], reverse=reverse
    )


def margin_rows(v: dict) -> list[dict]:
    out = []
    for (grid, sc), k, metric in itertools.product(CASES, CLASSES, METRICS):
        order = order_for(v, grid, sc, k, metric)
        for first, second in itertools.pairwise(order):
            a, b = (
                v[key(first, grid, sc, k)][metric],
                v[key(second, grid, sc, k)][metric],
            )
            diff = abs(b - a)
            ch_a, ch_b = changes(v, first, k, metric), changes(v, second, k, metric)
            applicable = [ch_a["grid"], ch_b["grid"]]
            if grid == PRODUCT_GRID:
                applicable += [ch_a["schmidt"], ch_b["schmidt"]]
            out.append(
                {
                    "grid": grid,
                    "sc": sc,
                    "class": k,
                    "diameter": v[key(first, grid, sc, k)]["diameter"],
                    "metric": metric,
                    "pair": [first, second],
                    "labels": [label_of(first, grid), label_of(second, grid)],
                    "values": [a, b],
                    "difference": diff,
                    "relative": diff / max(abs(a), abs(b))
                    if max(abs(a), abs(b)) > 0
                    else None,
                    "grid_change": [ch_a["grid"], ch_b["grid"]],
                    "schmidt_change": [ch_a["schmidt"], ch_b["schmidt"]]
                    if grid == PRODUCT_GRID
                    else None,
                    "robust": all(diff > c for c in applicable),
                }
            )
    return out


def robust_order(
    v: dict, margins: list[dict], grid: str, sc: float, k: str, metric: str
) -> str:
    """The case's order with tied neighbours grouped, best first."""
    order = order_for(v, grid, sc, k, metric)
    rows = [
        m
        for m in margins
        if (m["grid"], m["sc"], m["class"], m["metric"]) == (grid, sc, k, metric)
    ]
    text = label_of(order[0], grid)
    for layout, m in zip(order[1:], rows, strict=True):
        text += (ORDER if m["robust"] else TIE) + label_of(layout, grid)
    return text


def ranking_rows(v: dict, margins: list[dict]) -> list[dict]:
    out = []
    for (grid, sc), k in itertools.product(CASES, CLASSES):
        row = {
            "grid": grid,
            "sc": sc,
            "class": k,
            "diameter": v[key(LAYOUTS[0], grid, sc, k)]["diameter"],
        }
        for metric in METRICS:
            row[metric] = robust_order(v, margins, grid, sc, k, metric)
            row[f"{metric}_values"] = {
                label_of(lay, grid): v[key(lay, grid, sc, k)][metric] for lay in LAYOUTS
            }
        out.append(row)
    return out


def pair_relation(
    v: dict, grid: str, sc: float, k: str, metric: str, x: str, y: str
) -> str:
    """How x stands to y on one case and metric: "x > y", "y > x" or "x ~ y" (a tie)."""
    a, b = v[key(x, grid, sc, k)][metric], v[key(y, grid, sc, k)][metric]
    diff = abs(a - b)
    ch_x, ch_y = changes(v, x, k, metric), changes(v, y, k, metric)
    applicable = [ch_x["grid"], ch_y["grid"]]
    if grid == PRODUCT_GRID:
        applicable += [ch_x["schmidt"], ch_y["schmidt"]]
    if not all(diff > c for c in applicable):
        return f"{x} ~ {y}"
    better_is_lower = METRICS[metric]["better"] == "lower"
    x_ahead = (a < b) if better_is_lower else (a > b)
    return f"{x} > {y}" if x_ahead else f"{y} > {x}"


def summary_rows(v: dict, rankings: list[dict]) -> list[dict]:
    """Per metric and class: the order per case, and each pair's relation across the cases."""
    out = []
    for metric, k in itertools.product(METRICS, CLASSES):
        orders = [r[metric].replace("*", "") for r in rankings if r["class"] == k]
        same = len(set(orders)) == 1
        pairs = {}
        for x, y in itertools.combinations(LAYOUTS, 2):
            relations = [
                pair_relation(v, grid, sc, k, metric, x, y) for grid, sc in CASES
            ]
            counts = {rel: relations.count(rel) for rel in dict.fromkeys(relations)}
            pairs[f"{x} vs {y}"] = {
                "relations": relations,
                "text": "; ".join(
                    f"{rel} in {'all ' if n == len(CASES) else ''}{n} of {len(CASES)}"
                    for rel, n in counts.items()
                ),
            }
        out.append(
            {
                "metric": metric,
                "class": k,
                "diameter": next(r["diameter"] for r in rankings if r["class"] == k),
                "orders": orders,
                "one_order": same,
                "order": orders[0] if same else None,
                "has_tie": any(TIE in o for o in orders),
                "pairs": pairs,
            }
        )
    return out


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def fmt(x: float | None, digits: int = 4) -> str:
    return "-" if x is None else f"{x:.{digits}g}"


def um(d: float) -> str:
    return f"{d * 1e6:g}"


def table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def metric_table(v: dict) -> str:
    parts = [LEGEND, ""]
    for k in CLASSES:
        rows = []
        for layout in LAYOUTS:
            for grid, sc in CASES:
                e = v[key(layout, grid, sc, k)]
                stops = ", ".join(
                    f"{s} {e['stops'][s]}" for s in SOURCES if e["stops"][s] != "steady"
                )
                rows.append(
                    [
                        f"{e['label']}, {grid}, {sc_key(sc)}",
                        ", ".join(fmt(e["per_source"][s]["M1"]) for s in SOURCES)
                        + f"; {fmt(e['M1'])}",
                        ", ".join(
                            f"{fmt(e['per_source'][s]['M2_terms']['near_door'])} + "
                            f"{fmt(e['per_source'][s]['M2_terms']['hood_entry'])}"
                            for s in SOURCES
                        )
                        + f"; {fmt(e['M2'])}",
                        ", ".join(fmt(e["per_source"][s]["M3"], 5) for s in SOURCES)
                        + f"; {fmt(e['M3'], 5)}",
                        stops or "all steady",
                    ]
                )
        d = v[key(LAYOUTS[0], GRIDS[0], BASE_SC, k)]["diameter"]
        parts.append(
            f"The metrics, {um(d)} um (M1 particles per s per m depth, lower better; M2 per m^3 "
            "per (particle per s per m), lower better; M3 a share, higher better):"
        )
        parts.append(
            table(
                [
                    "Layout, grid, Sc_t",
                    "M1: S1, S2, S3; sum",
                    "M2: near_door + hood_entry for S1, S2, S3; sum",
                    "M3: S1, S2, S3; mean",
                    "Marches not steady",
                ],
                rows,
            )
        )
        parts.append("")
    return "\n".join(parts).rstrip()


def ranking_table(rankings: list[dict]) -> str:
    rows = [
        [
            f"{r['grid']}, {sc_key(r['sc'])}, {um(r['diameter'])}",
            r["M1"],
            r["M2"],
            r["M3"],
        ]
        for r in rankings
    ]
    return (
        LEGEND
        + "\n\nBest first; `A > B` means A ranks ahead of B robustly (the difference exceeds "
        "every applicable change), `A ~ B` a tie.\n\n"
        + table(
            [
                "Grid, Sc_t, class (um)",
                "M1 process-top deposition (lower better)",
                "M2 operator exposure (lower better)",
                "M3 removal (higher better)",
            ],
            rows,
        )
    )


def margin_table(margins: list[dict]) -> str:
    rows = [
        [
            f"{m['grid']}, {sc_key(m['sc'])}, {um(m['diameter'])}",
            m["metric"],
            ", ".join(m["labels"]),
            ", ".join(fmt(x, 5 if m["metric"] == "M3" else 4) for x in m["values"]),
            f"{fmt(m['difference'])}; {fmt(m['relative'], 3)}",
            ", ".join(fmt(c, 3) for c in m["grid_change"]),
            ", ".join(fmt(c, 3) for c in m["schmidt_change"])
            if m["schmidt_change"]
            else "-",
            "yes" if m["robust"] else "no (tie)",
        ]
        for m in margins
    ]
    return table(
        [
            "Grid, Sc_t, class (um)",
            "Metric",
            "Pair (better, worse)",
            "Values",
            "Difference; relative to the larger",
            "Grid's change to each (at Sc_t 0.7)",
            "Sc_t's change to each (320x120, largest of 0.2 and 1.3 against 0.7)",
            "Robust",
        ],
        rows,
    )


def summary_table(summary: list[dict]) -> str:
    rows = [
        [
            f"{s['metric']} {METRICS[s['metric']]['name']} ({METRICS[s['metric']]['better']} better)",
            um(s["diameter"]),
            s["pairs"]["L0 vs L1"]["text"],
            s["pairs"]["L0 vs L2"]["text"],
            s["pairs"]["L1 vs L2"]["text"],
            s["order"] if s["one_order"] else "no; " + "; ".join(s["orders"]),
        ]
        for s in summary
    ]
    return (
        "The L2 rows are the starred supplement in every case (the star is dropped here). "
        "Each pair's relation is computed on each of the four cases by the same rule as the "
        "margin table (`X > Y`: X ranks ahead of Y robustly; `X ~ Y`: a tie).\n\n"
        + table(
            [
                "Metric",
                "Class (um)",
                "L0 against L1",
                "L0 against L2",
                "L1 against L2",
                "One robust order in every case",
            ],
            rows,
        )
    )


def render() -> tuple[str, dict]:
    v = values()
    margins = margin_rows(v)
    rankings = ranking_rows(v, margins)
    summary = summary_rows(v, rankings)
    md = "\n\n".join(
        [
            "### metrics\n\n" + metric_table(v),
            "### ranking\n\n" + ranking_table(rankings),
            "### margins\n\n" + margin_table(margins),
            "### summary\n\n" + summary_table(summary),
        ]
    )
    data = {"values": v, "margins": margins, "rankings": rankings, "summary": summary}
    return md + "\n", data


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

COLOURS = {"L0": "#2a78d6", "L1": "#eb6834", "L2": "#1baf7a"}
MARKERS = {
    ("160x60", 0.7): "o",
    ("320x120", 0.2): "s",
    ("320x120", 0.7): "D",
    ("320x120", 1.3): "^",
}
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID_LINE = "#e3e2de"


def figure(v: dict, rankings: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(len(CLASSES), len(METRICS), figsize=(13.5, 8.0), dpi=150)
    xs = {lay: i for i, lay in enumerate(LAYOUTS)}
    for row, k in enumerate(CLASSES):
        for col, metric in enumerate(METRICS):
            ax = axes[row][col]
            for (grid, sc), marker in MARKERS.items():
                for lay in LAYOUTS:
                    e = v[key(lay, grid, sc, k)]
                    ax.plot(
                        xs[lay],
                        e[metric],
                        marker=marker,
                        markersize=8,
                        markerfacecolor=COLOURS[lay],
                        markeredgecolor="white",
                        markeredgewidth=1.0,
                        linestyle="none",
                        alpha=0.95,
                    )
            if metric in ("M1", "M2"):
                ax.set_yscale("log")
            ax.set_xticks(list(xs.values()))
            ax.set_xticklabels(
                [label_of(lay, PRODUCT_GRID) for lay in LAYOUTS], color=INK
            )
            ax.set_xlim(-0.6, len(LAYOUTS) - 0.4)
            ax.grid(axis="y", color=GRID_LINE, linewidth=0.6)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(GRID_LINE)
            ax.tick_params(colors=INK_SECONDARY, labelsize=8)
            d = v[key(LAYOUTS[0], GRIDS[0], BASE_SC, k)]["diameter"]
            info = METRICS[metric]
            ax.set_title(
                f"{metric} {info['name']} ({info['better']} better), {um(d)} um",
                fontsize=9.5,
                color=INK,
                loc="left",
            )
            product = next(
                r
                for r in rankings
                if (r["grid"], r["sc"], r["class"]) == (PRODUCT_GRID, BASE_SC, k)
            )
            orders = {r[metric] for r in rankings if r["class"] == k}
            text = f"robust order, 320x120 at Sc_t 0.7:\n{product[metric]}"
            if len(orders) > 1:
                text += "  (differs by case: the ranking table)"
            ax.text(
                0.0,
                -0.16,
                text,
                transform=ax.transAxes,
                fontsize=8,
                color=INK_SECONDARY,
                ha="left",
                va="top",
            )
    handles = [
        Line2D(
            [0],
            [0],
            marker=m,
            linestyle="none",
            markersize=7,
            markerfacecolor=INK_SECONDARY,
            markeredgecolor="white",
            label=f"{grid}, Sc_t {sc_key(sc)}",
        )
        for (grid, sc), m in MARKERS.items()
    ] + [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markersize=7,
            markerfacecolor=COLOURS[lay],
            markeredgecolor="white",
            label=label_of(lay, PRODUCT_GRID),
        )
        for lay in LAYOUTS
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(handles),
        frameon=False,
        fontsize=8,
        labelcolor=INK,
    )
    fig.suptitle(
        "Phase 3 gate: the three layouts against the three metrics (step 8's records, prompt 58). "
        "`A > B` A ranks ahead of B robustly, `~` a tie; L2* is step 8's labelled supplement.",
        fontsize=10,
        color=INK,
        x=0.01,
        ha="left",
    )
    fig.patch.set_facecolor("#fcfcfb")
    for ax in axes.flat:
        ax.set_facecolor("#fcfcfb")
    fig.tight_layout(rect=(0, 0.06, 1, 0.95), h_pad=3.4, w_pad=2.0)
    fig.savefig(FIGURE, facecolor=fig.get_facecolor())
    print(f"wrote {FIGURE.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def write_tables(md: str, data: dict) -> None:
    TABLES_MD.write_text(md, encoding="utf-8", newline="\n")
    TABLES_JSON.write_text(
        json.dumps(data, indent=1, default=float) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"wrote {TABLES_MD.relative_to(ROOT)} and {TABLES_JSON.relative_to(ROOT)}")


def check(md: str, data: dict) -> int:
    """Compare the regenerated tables with the committed files; 0 when equal."""
    ok = True
    committed_md = TABLES_MD.read_text(encoding="utf-8").replace("\r\n", "\n")
    if committed_md != md:
        ok = False
        print("tables58.md differs from the regenerated tables")
    committed = json.loads(TABLES_JSON.read_text(encoding="utf-8"))
    regenerated = json.loads(json.dumps(data, default=float))
    if committed != regenerated:
        ok = False
        print("tables58.json differs from the regenerated values")
    print("check: " + ("the committed tables reproduce" if ok else "DIFFERENCES FOUND"))
    return 0 if ok else 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "mode", nargs="?", choices=["tables", "figure", "all"], default="all"
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    md, data = render()
    if args.check:
        sys.exit(check(md, data))
    if args.mode in ("tables", "all"):
        print(md)
        write_tables(md, data)
    if args.mode in ("figure", "all"):
        figure(data["values"], data["rankings"])


if __name__ == "__main__":
    main()
