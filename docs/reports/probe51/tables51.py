"""The tables of docs/reports/turbulence_cg_solve.md, generated from prompt 51's records.

Usage:
    python tables51.py

Reads results/builder51/ of the main checkout (the records check51.py,
plant51.py, annex51.py, product51.py and val016_51.py write) and replaces
each region of the report between ``<!-- BEGIN tables51: NAME -->`` and
``<!-- END tables51: NAME -->`` with the table NAME, so no number in the
tables is typed by hand. A region whose records are missing is left as it
was and named on stdout.
"""

import json
import re
import subprocess
from collections.abc import Callable
from pathlib import Path

HERE = Path(__file__).resolve().parents[3]
MAIN = Path(
    subprocess.run(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
        cwd=HERE,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
).parent
OUT = MAIN / "results" / "builder51"
REPORT = HERE / "docs" / "reports" / "turbulence_cg_solve.md"
GRIDS = ("80x30", "160x60")
# Prompt 49's whole-run Jacobi means on the Annex 20 row, the prompt's premise.
PROMPT_JACOBI = {"k": 272, "eps": 182}


def load(name: str) -> dict | list:
    """One record of results/builder51/."""
    return json.loads((OUT / name).read_text())


def row(*cells: object) -> str:
    """One markdown table row, a pipe inside a cell escaped."""
    return "| " + " | ".join(str(c).replace("|", "\\|") for c in cells) + " |"


def table(header: list[str], rows: list[list[object]]) -> str:
    """A markdown table."""
    lines = [row(*header), row(*["---"] * len(header))]
    lines += [row(*r) for r in rows]
    return "\n".join(lines)


def stop() -> str:
    """The first pass's stop: the k system at chosen steps of the long standard run."""
    rows = []
    for scheme in ("upwind", "umist"):
        rec = load(f"stop_{scheme}.json")
        failed = rec["failure"]["step"] if rec["failure"] else None
        for step, s in rec["steps"].items():
            label = (
                f"{step} (raises)"
                if failed is not None and int(step) == failed
                else step
            )
            rows.append(
                [
                    scheme,
                    label,
                    f"{s['conductance_over_v_dt']:.1e}",
                    f"{s['condition']:.1e}",
                    f"{s['dense_min']:.3e}",
                    f"{s['jacobi'][2]:.3e} ({s['jacobi'][0]}, "
                    f"{'conv' if s['jacobi'][1] else 'cap'})",
                    f"{s['cg_no_fallback'][2]:.3e} ({s['cg_no_fallback'][0]}, "
                    f"{'conv' if s['cg_no_fallback'][1] else 'cap'})",
                    f"{s['cg_committed'][2]:.3e} (fallback {s['cg_committed'][1]})",
                ]
            )
    return table(
        [
            "Scheme",
            "Step",
            "Largest G / smallest V/dt",
            "Condition number",
            "Dense solve, least k",
            "Jacobi, least k (its, stop)",
            "CG without fallback, least k (its, stop)",
            "Committed CG, least k",
        ],
        rows,
    )


def item4() -> str:
    """Item 4: the laminar hashes and the Jacobi battery, base against commit A."""
    report = load("report_base_A.json")
    rows = []
    for entry in report:
        if entry["case"] == "jacobi_battery":
            rows.append(
                [
                    "Jacobi battery (240 systems)",
                    f"{entry['capped'][0]} capped / {entry['capped'][1]} capped",
                    f"{entry['sweeps_total'][0]:,} / {entry['sweeps_total'][1]:,} sweeps",
                    f"`{entry['hash'][0]}` / `{entry['hash'][1]}`",
                    "yes" if entry["equal"] else "NO",
                ]
            )
        else:
            rows.append(
                [
                    entry["case"],
                    f"{entry['stop'][0]} / {entry['stop'][1]}",
                    f"{entry['outer'][0]:,} / {entry['outer'][1]:,}",
                    f"`{entry['face_hash'][0]}` / `{entry['face_hash'][1]}`",
                    "yes" if entry["equal"] else "NO",
                ]
            )
    return table(
        [
            "Case",
            "Stop (main / A)",
            "Count (main / A)",
            "Hash, 16 hex (main / A)",
            "Bitwise",
        ],
        rows,
    )


def plants() -> str:
    """Commit A's planted defects and the tests that caught each."""
    rows = []
    for p in load("plants.json"):
        caught = [f.split("::", 1)[1] for f in p["failures"]]
        shown = "; ".join(f"`{c}`" for c in caught[:3])
        if len(caught) > 3:
            shown += f"; and {len(caught) - 3} more"
        rows.append(
            [
                f"`{p['plant']}`",
                p["meaning"],
                "caught" if p["caught"] else "SURVIVED",
                shown or "none",
            ]
        )
    return table(["Plant", "What it stands for", "Outcome", "Failing tests"], rows)


def annex_cost() -> str:
    """Item 5: the Annex 20 row on main and on the branch."""
    c = load("compare_main_cg.json")
    rows = []
    for key, name in (("base", "main (Jacobi)"), ("built", "branch (CG)")):
        s = c[key]
        lines = s["lines"]
        rows.append(
            [
                f"{name}, `{s['commit']}`",
                s["stop"],
                f"{s['outer']:,}",
                f"{s['wall_seconds']:,.0f}",
                f"{s['seconds_per_outer']:.4f}",
                f"{s['k_iterations_mean']:.1f}",
                f"{s['eps_iterations_mean']:.1f}",
                f"{s['pressure_iterations_mean']:.1f}",
                s["pressure_cap_hits"],
                s["fallbacks"],
                f"{lines['x3']['rms']:.4f} / {lines['x3']['allowance']:.4f}",
                f"{lines['y2916']['rms']:.4f} / {lines['y2916']['allowance']:.4f}",
                "pass" if s["passed"] else "FAIL",
            ]
        )
    return table(
        [
            "Row",
            "Stop",
            "Outer",
            "Wall s",
            "s per outer",
            "k its per solve",
            "eps its per solve",
            "CG its per p' solve",
            "p' caps",
            "Fallbacks",
            "x3 RMS / allowance",
            "y2916 RMS / allowance",
            "Scored lines",
        ],
        rows,
    )


def annex_stages() -> str:
    """Item 5: the harness's stage seconds on both Annex 20 rows, and the share of each."""
    recs = {lab: load(f"annex_{lab}.json")["row"] for lab in ("main", "cg")}
    stages = list(recs["main"]["time"]["stages"])
    rows = []
    for stage in stages:
        cells = [stage]
        for lab in ("main", "cg"):
            r = recs[lab]
            seconds = r["time"]["stages"][stage]
            outer = r["work"]["outer_iterations"]
            share = seconds / r["time"]["wall_seconds"]
            cells += [f"{seconds:,.0f}", f"{seconds / outer:.4f}", f"{share:.1%}"]
        before = recs["main"]["time"]["stages"][stage]
        cells.append(f"{(recs['cg']['time']['stages'][stage] - before) / before:+.1%}")
        rows.append(cells)
    main_wall = recs["main"]["time"]["wall_seconds"]
    turbulence = recs["main"]["time"]["stages"]["turbulence"]
    needed = 0.40 * main_wall / turbulence
    rows.append(
        [
            "the turbulence stage's fall that a 40% fall of the whole run needs",
            "",
            "",
            "",
            "",
            "",
            "",
            f"-{needed:.0%}",
        ]
    )
    return table(
        [
            "Stage",
            "Main s",
            "Main s per outer",
            "Main share",
            "Branch s",
            "Branch s per outer",
            "Branch share",
            "Change",
        ],
        rows,
    )


def loop() -> str:
    """One iteration of each loop on the Annex 20 mesh, isolated, by part."""
    rec = load("loop.json")
    rows = [
        ["Jacobi sweep", f"{rec['jacobi_sweep_ms']:.4f}"],
        ["CG iteration", f"{rec['cg_iteration_ms']:.4f}"],
    ]
    for part, ms in rec["parts_ms"].items():
        rows.append([f"CG part: {part.replace('_', ' ')}", f"{ms:.4f}"])
    rows.append(
        [
            "CG iteration over Jacobi sweep",
            f"{rec['cg_iteration_ms'] / rec['jacobi_sweep_ms']:.2f}",
        ]
    )
    return table(["Per iteration, 216 x 72, one thread", "ms"], rows)


def annex_answers() -> str:
    """Item 6: the Annex 20 rows' differences."""
    c = load("compare_main_cg.json")
    rows = [
        ["Outer count, branch against main", f"{c['outer_change']:+.2%}"],
        [
            "Seconds per outer iteration, branch against main",
            f"{c['seconds_per_outer_change']:+.1%}",
        ],
    ]
    for key, value in c["line_difference_u0"].items():
        rows.append([f"Largest difference on line {key}, u / u0", f"{value:.2e}"])
    rows.append(
        ["Largest line difference, u / u0", f"{c['largest_line_difference_u0']:.2e}"]
    )
    return table(["Quantity", "Value"], rows)


def products() -> str:
    """Item 6: prompt 48's product rows on main and on the branch, against prompt 50's."""
    rows = []
    for grid in GRIDS:
        c = load(f"compare_product_{grid}.json")
        for lab, name in (("main", "main (Jacobi)"), ("cg", "branch (CG)")):
            s = c[lab]
            rows.append(
                [
                    grid,
                    f"{name}, `{s['commit']}`",
                    s["stop"],
                    f"{s['outer']:,}",
                    f"{c['prompt50']['outer']:,}",
                    f"{s['wall_seconds']:.0f}",
                    f"{s['seconds_per_outer']:.4f}",
                    f"{s['k_iterations_mean']:.1f}",
                    f"{s['eps_iterations_mean']:.1f}",
                    s["fallbacks"],
                    s["scalar_cap_hits"],
                ]
            )
    return table(
        [
            "Grid",
            "Row",
            "Stop",
            "Outer",
            "Prompt 50's outer",
            "Wall s",
            "s per outer",
            "k its per solve",
            "eps its per solve",
            "Fallbacks",
            "k/eps caps",
        ],
        rows,
    )


def product_faces() -> str:
    """Item 6: the product rows' face differences over the velocity scale."""
    rows = []
    for grid in GRIDS:
        c = load(f"compare_product_{grid}.json")
        rows.append(
            [
                grid,
                f"{c['velocity_scale']:g}",
                "yes" if c["base_equals_prompt50_faces"] else "NO",
                f"{c['face_difference_over_scale_built_vs_base']:.2e}",
                f"{c['face_difference_over_scale_built_vs_prompt50']:.2e}",
                f"{c['seconds_per_outer_change']:+.1%}",
            ]
        )
    return table(
        [
            "Grid",
            "Velocity scale, m/s",
            "Main's faces bitwise prompt 50's",
            "Largest face difference, branch vs main, over the scale",
            "Same, branch vs prompt 50",
            "s per outer, branch vs main",
        ],
        rows,
    )


def val016() -> str:
    """Item 6: VAL-016's criteria on main and on the branch."""
    rows = []
    for lab, name in (("main", "main (Jacobi)"), ("cg", "branch (CG)")):
        r = load(f"val016_{lab}.json")
        core = r["b_core_k_ratio"]
        rows.append(
            [
                f"{name}, `{r['commit']}`",
                r["stop"],
                f"{r['outer']:,}",
                f"{r['k_iterations_mean']:.1f} / {r['eps_iterations_mean']:.1f}",
                r["fallbacks"],
                f"{r['a_du_over_uw']:.2e}",
                f"{r['a_dk_over_utau2']:.2e}",
                f"{r['a_u_tau_ratio']:.6f}",
                f"{core['standard'][0]:.5f} to {core['standard'][1]:.5f}",
                f"{core['rng'][0]:.5f} to {core['rng'][1]:.5f}",
            ]
        )
    return table(
        [
            "Tree",
            "Stop",
            "Outer",
            "k / eps its per solve",
            "Fallbacks",
            "(a) max du / U_w (< 1e-4)",
            "(a) max dk / u_tau^2 (< 1e-4)",
            "(a) u_tau / 1D",
            "(b) standard core k ratio",
            "(b) RNG core k ratio",
        ],
        rows,
    )


def predictions() -> str:
    """The predictions against the measurements."""
    annex = load("compare_main_cg.json")
    base, built = annex["base"], annex["built"]
    report4 = load("report_base_A.json")
    plants_all = load("plants.json")
    prods = {g: load(f"compare_product_{g}.json") for g in GRIDS}
    v016 = load("val016_cg.json")
    a_ok = all(e["equal"] for e in report4)
    b_val = max(built["k_iterations_mean"], built["eps_iterations_mean"])
    c_val = -annex["seconds_per_outer_change"]
    d_outer = abs(annex["outer_change"])
    d_lines = annex["largest_line_difference_u0"]
    d_ok = d_outer <= 0.02 and built["passed"] and d_lines < 1e-4
    positivity = (
        [built.get("stop")] + [prods[g]["cg"]["stop"] for g in GRIDS] + [v016["stop"]]
    )
    e_ok = "positivity_error" not in positivity
    fallbacks = built["fallbacks"] + sum(prods[g]["cg"]["fallbacks"] for g in GRIDS)
    rows = [
        [
            "(a)",
            "laminar hashes, pressure loop and transport gate unchanged bit for bit",
            "all three laminar cases and the 240-system Jacobi battery bitwise main's"
            if a_ok
            else "a case differs",
            "holds" if a_ok else "FAILS",
        ],
        [
            "(b)",
            f"mean CG iterations per k or eps solve under 60 (Jacobi {PROMPT_JACOBI['k']} and "
            f"{PROMPT_JACOBI['eps']})",
            f"k {built['k_iterations_mean']:.1f}, eps {built['eps_iterations_mean']:.1f} "
            f"(this run's Jacobi: {base['k_iterations_mean']:.1f}, {base['eps_iterations_mean']:.1f})",
            "holds" if b_val < 60 else "FAILS",
        ],
        [
            "(c)",
            "seconds per outer iteration fall by at least 40%",
            f"{c_val:.1%} lower",
            "holds" if c_val >= 0.40 else "FAILS",
        ],
        [
            "(d)",
            "stop within 2% of main's count, both scored lines pass, lines within 1e-4 u0",
            f"count {annex['outer_change']:+.2%}, scored lines "
            f"{'pass' if built['passed'] else 'FAIL'}, largest line difference {d_lines:.1e} u0",
            "holds" if d_ok else "FAILS",
        ],
        [
            "(e)",
            "no positivity error in any run",
            "none" if e_ok else "a positivity error",
            "holds" if e_ok else "FAILS",
        ],
        [
            "(f)",
            "zero fallbacks on the Annex 20 and product rows",
            f"{fallbacks}",
            "holds" if fallbacks == 0 else "FAILS",
        ],
    ]
    caught = sum(p["caught"] for p in plants_all)
    rows.append(
        [
            "plants",
            "every planted defect fails a test",
            f"{caught} of {len(plants_all)}",
            "",
        ]
    )
    return table(["", "Prediction", "Measured", "Verdict"], rows)


TABLES: dict[str, Callable[[], str]] = {
    "stop": stop,
    "item4": item4,
    "plants": plants,
    "annex_cost": annex_cost,
    "annex_stages": annex_stages,
    "loop": loop,
    "annex_answers": annex_answers,
    "products": products,
    "product_faces": product_faces,
    "val016": val016,
    "predictions": predictions,
}


def main() -> None:
    """Splice every table whose records exist into the report."""
    raw = REPORT.read_bytes().decode()
    crlf = "\r\n" in raw
    text = raw.replace("\r\n", "\n")
    for name, build in TABLES.items():
        pattern = re.compile(
            rf"(<!-- BEGIN tables51: {name} -->\n).*?(<!-- END tables51: {name} -->)",
            re.DOTALL,
        )
        if not pattern.search(text):
            print(f"{name}: no region in the report")
            continue
        try:
            body = build()
        except FileNotFoundError as missing:
            print(f"{name}: records missing ({missing.filename})")
            continue
        text = pattern.sub(lambda m, b=body: m.group(1) + b + "\n" + m.group(2), text)
        print(f"{name}: written")
    if crlf:
        text = text.replace("\n", "\r\n")
    REPORT.write_bytes(text.encode())


if __name__ == "__main__":
    main()
